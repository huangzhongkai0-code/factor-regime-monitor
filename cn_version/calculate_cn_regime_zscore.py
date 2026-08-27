"""根据中国 A 股 ETF 的滚动年化收益率计算 Z-score 和状态标签。

默认输入：cn_rolling_returns.csv
默认输出：
- cn_regime_zscore.csv：数值型 Z-score
- cn_regime_labels.csv：Expansion / Neutral / Contraction 状态标签

当前值与“此前”252个交易日的均值和样本标准差比较。历史窗口先使用
shift(1) 滞后一日，确保当前值不会进入自身的基准统计，避免未来数据泄漏。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    missing_package = getattr(exc, "name", "pandas/numpy")
    raise SystemExit(
        f"缺少依赖 {missing_package!r}。请先运行：python -m pip install pandas numpy"
    ) from exc


ETF_CODES = [
    "510300",
    "510880",
    "159915",
    "512100",
    "159928",
    "588000",
    "512010",
    "512480",
    "512660",
    "512880",
    "515030",
    "512800",
]

REFERENCE_WINDOW = 252
DEFAULT_UPPER_THRESHOLD = 1.0
DEFAULT_LOWER_THRESHOLD = -1.0

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "cn_rolling_returns.csv"
ZSCORE_FILENAME = "cn_regime_zscore.csv"
LABELS_FILENAME = "cn_regime_labels.csv"


class DataValidationError(ValueError):
    """输入数据或参数不满足计算要求。"""


def load_and_validate_rolling_returns(input_path: Path) -> pd.DataFrame:
    """读取滚动收益率 CSV，并校验日期索引、列和数值类型。"""

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    try:
        raw_returns = pd.read_csv(
            input_path,
            index_col="Date",
            parse_dates=["Date"],
            encoding="utf-8-sig",
        )
    except ValueError as error:
        raise DataValidationError(
            "无法按 Date 索引读取输入 CSV，请检查表头和日期列。"
        ) from error

    missing_codes = [code for code in ETF_CODES if code not in raw_returns.columns]
    if missing_codes:
        raise DataValidationError("输入文件缺少 ETF 列：" + ", ".join(missing_codes))

    rolling_returns = raw_returns.reindex(columns=ETF_CODES).copy()
    parsed_index = pd.to_datetime(rolling_returns.index, errors="coerce")
    if parsed_index.isna().any():
        raise DataValidationError("Date 索引包含无法识别的日期。")
    rolling_returns.index = parsed_index
    rolling_returns = rolling_returns.sort_index()

    if rolling_returns.index.has_duplicates:
        duplicated_date = rolling_returns.index[
            rolling_returns.index.duplicated()
        ][0]
        raise DataValidationError(
            f"Date 索引存在重复日期，例如：{duplicated_date:%Y-%m-%d}"
        )

    # 上市较晚或滚动窗口不足形成的 NaN 属正常；只拦截无法解析的非空文本。
    numeric_returns = rolling_returns.apply(pd.to_numeric, errors="coerce")
    invalid_numeric_mask = rolling_returns.notna() & numeric_returns.isna()
    if invalid_numeric_mask.any().any():
        row_position, column_position = np.argwhere(
            invalid_numeric_mask.to_numpy()
        )[0]
        invalid_date = rolling_returns.index[row_position]
        invalid_code = rolling_returns.columns[column_position]
        raise DataValidationError(
            f"{invalid_code} 在 {invalid_date:%Y-%m-%d} 包含无法解析的数值。"
        )

    infinite_mask = pd.DataFrame(
        np.isinf(numeric_returns.to_numpy(dtype=float)),
        index=numeric_returns.index,
        columns=numeric_returns.columns,
    )
    if infinite_mask.any().any():
        row_position, column_position = np.argwhere(infinite_mask.to_numpy())[0]
        invalid_date = numeric_returns.index[row_position]
        invalid_code = numeric_returns.columns[column_position]
        raise DataValidationError(
            f"{invalid_code} 在 {invalid_date:%Y-%m-%d} 包含无穷值。"
        )

    if numeric_returns.empty or not numeric_returns.notna().any().any():
        raise DataValidationError("输入文件为空或没有任何有效滚动收益率。")

    numeric_returns.index.name = "Date"
    numeric_returns.columns.name = None
    return numeric_returns.astype(float)


def calculate_zscores(rolling_returns: pd.DataFrame) -> pd.DataFrame:
    """逐列计算当前值相对此前252个交易日分布的 Z-score。"""

    # 关键防泄漏步骤：当前行先滞后一日，再用于历史均值和标准差。
    historical_values = rolling_returns.shift(1)
    historical_mean = historical_values.rolling(
        window=REFERENCE_WINDOW,
        min_periods=REFERENCE_WINDOW,
    ).mean()
    historical_standard_deviation = historical_values.rolling(
        window=REFERENCE_WINDOW,
        min_periods=REFERENCE_WINDOW,
    ).std(ddof=1)

    # 标准差等于 0 时 Z-score 无定义，保留 NaN 而不是生成无穷值。
    valid_standard_deviation = historical_standard_deviation.where(
        historical_standard_deviation != 0
    )
    zscores = (rolling_returns - historical_mean) / valid_standard_deviation
    zscores.index.name = "Date"
    zscores.columns.name = None
    return zscores


def classify_regimes(
    zscores: pd.DataFrame,
    upper_threshold: float,
    lower_threshold: float,
) -> pd.DataFrame:
    """按照可调阈值把有效 Z-score 分成三种状态。"""

    if not math.isfinite(upper_threshold) or not math.isfinite(lower_threshold):
        raise DataValidationError("上下阈值必须是有限数字。")
    if lower_threshold >= upper_threshold:
        raise DataValidationError("下阈值必须严格小于上阈值。")

    # 历史窗口不足时标签保持为空；只有有效 Z-score 才进行分类。
    labels = pd.DataFrame(
        pd.NA,
        index=zscores.index,
        columns=zscores.columns,
        dtype="string",
    )
    valid_zscores = zscores.notna()
    labels = labels.mask(valid_zscores, "Neutral")
    labels = labels.mask(zscores > upper_threshold, "Expansion")
    labels = labels.mask(zscores < lower_threshold, "Contraction")
    labels.index.name = "Date"
    labels.columns.name = None
    return labels


def save_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """先写临时文件再替换正式文件，避免留下不完整 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(
            temporary_path,
            index=True,
            index_label="Date",
            date_format="%Y-%m-%d",
            encoding="utf-8-sig",
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def format_date(date_value: pd.Timestamp | None) -> str:
    """把日期或空值格式化为终端字符串。"""

    return "无" if date_value is None else date_value.strftime("%Y-%m-%d")


def format_zscore(value: object) -> str:
    """把最新 Z-score 格式化为四位小数。"""

    return "NaN" if pd.isna(value) else f"{float(value):.4f}"


def format_label(value: object) -> str:
    """把空标签显示为数据不足。"""

    return "数据不足" if pd.isna(value) else str(value)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"滚动收益率 CSV（默认：{DEFAULT_INPUT_PATH}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIRECTORY,
        help=f"两个结果 CSV 的保存目录（默认：{SCRIPT_DIRECTORY}）",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=DEFAULT_UPPER_THRESHOLD,
        help=f"Expansion 上阈值（默认：{DEFAULT_UPPER_THRESHOLD:g}）",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=DEFAULT_LOWER_THRESHOLD,
        help=f"Contraction 下阈值（默认：{DEFAULT_LOWER_THRESHOLD:g}）",
    )
    return parser.parse_args()


def main() -> int:
    """程序入口：读取数据、计算 Z-score、分类并保存。"""

    arguments = parse_arguments()
    input_path = arguments.input.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    zscore_path = output_directory / ZSCORE_FILENAME
    labels_path = output_directory / LABELS_FILENAME

    try:
        rolling_returns = load_and_validate_rolling_returns(input_path)
        zscores = calculate_zscores(rolling_returns)
        regime_labels = classify_regimes(
            zscores,
            upper_threshold=arguments.upper_threshold,
            lower_threshold=arguments.lower_threshold,
        )
        save_csv_atomically(zscores, zscore_path)
        save_csv_atomically(regime_labels, labels_path)
    except (
        FileNotFoundError,
        OSError,
        DataValidationError,
        pd.errors.ParserError,
    ) as error:
        print(f"[错误] {error}", file=sys.stderr)
        return 1

    latest_date = zscores.index[-1]
    latest_zscores = zscores.iloc[-1]
    latest_labels = regime_labels.iloc[-1]

    print(
        f"阈值：Expansion > {arguments.upper_threshold:g}；"
        f"Contraction < {arguments.lower_threshold:g}；其余为 Neutral。"
    )
    print(
        f"历史基准：严格使用 shift(1) 后的过去 {REFERENCE_WINDOW} 个交易日，"
        "当前值不参与自身基准统计。"
    )
    print(f"{ZSCORE_FILENAME}：{zscores.shape[0]} 行 × {zscores.shape[1]} 列")
    print(
        f"{LABELS_FILENAME}："
        f"{regime_labels.shape[0]} 行 × {regime_labels.shape[1]} 列"
    )

    print("\n每只 ETF 的首个有效 Z-score 日期：")
    for code in ETF_CODES:
        print(f"  {code}：{format_date(zscores[code].first_valid_index())}")

    print(f"\n最新一期：{latest_date:%Y-%m-%d}")
    for code in ETF_CODES:
        print(
            f"  {code}：Z-score={format_zscore(latest_zscores[code])}，"
            f"状态={format_label(latest_labels[code])}"
        )

    print(f"结果已保存至：{output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
