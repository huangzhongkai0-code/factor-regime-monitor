"""根据滚动年化收益率计算 Z-score，并判断因子状态。

默认输入：rolling_returns.csv
默认输出：
- regime_zscore.csv：数值型 Z-score
- regime_labels.csv：Expansion / Contraction / Neutral 状态标签

计算约定：当前值与“此前”252个交易日的均值和样本标准差比较，
因此参考窗口会先滞后一日，当前值不会参与自身的基准统计。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("缺少 pandas。请先运行：python -m pip install pandas") from exc


ETF_TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
REFERENCE_WINDOW = 252
DEFAULT_UPPER_THRESHOLD = 1.0
DEFAULT_LOWER_THRESHOLD = -1.0

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "rolling_returns.csv"
ZSCORE_FILENAME = "regime_zscore.csv"
LABELS_FILENAME = "regime_labels.csv"


class DataValidationError(ValueError):
    """输入数据或参数不满足计算要求。"""


def load_and_validate_rolling_returns(input_path: Path) -> pd.DataFrame:
    """读取滚动收益率 CSV，并校验日期索引和 ETF 列。"""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    rolling_returns = pd.read_csv(
        input_path,
        index_col="Date",
        parse_dates=["Date"],
    )

    missing_tickers = [
        ticker for ticker in ETF_TICKERS if ticker not in rolling_returns.columns
    ]
    if missing_tickers:
        raise DataValidationError("输入文件缺少列：" + ", ".join(missing_tickers))

    # 只保留要求的 ETF；窗口期形成的空值继续保留，不做填充。
    rolling_returns = rolling_returns.reindex(columns=ETF_TICKERS).apply(
        pd.to_numeric,
        errors="coerce",
    )
    rolling_returns.index = pd.to_datetime(rolling_returns.index, errors="coerce")
    rolling_returns = rolling_returns.loc[~rolling_returns.index.isna()].sort_index()
    rolling_returns.index.name = "Date"
    rolling_returns.columns.name = None

    if rolling_returns.empty:
        raise DataValidationError("输入文件没有可用数据。")
    if rolling_returns.index.duplicated().any():
        duplicated_dates = rolling_returns.index[
            rolling_returns.index.duplicated()
        ].strftime("%Y-%m-%d")
        raise DataValidationError("日期索引存在重复值，例如：" + duplicated_dates[0])

    empty_tickers = [
        ticker for ticker in ETF_TICKERS if rolling_returns[ticker].notna().sum() == 0
    ]
    if empty_tickers:
        raise DataValidationError("以下列完全没有有效数值：" + ", ".join(empty_tickers))

    return rolling_returns.astype(float)


def calculate_zscores(rolling_returns: pd.DataFrame) -> pd.DataFrame:
    """计算当前滚动收益率相对此前 252 个交易日分布的 Z-score。"""
    # shift(1) 确保当前值不会进入自身的历史均值和标准差，避免信息泄漏。
    historical_values = rolling_returns.shift(1)
    historical_mean = historical_values.rolling(
        window=REFERENCE_WINDOW,
        min_periods=REFERENCE_WINDOW,
    ).mean()
    historical_standard_deviation = historical_values.rolling(
        window=REFERENCE_WINDOW,
        min_periods=REFERENCE_WINDOW,
    ).std(ddof=1)

    # 标准差为 0 时 Z-score 无定义，将分母置空以保留 NaN，而不是产生无穷值。
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
    """使用可调阈值把 Z-score 分类为三种状态。"""
    if lower_threshold >= upper_threshold:
        raise DataValidationError("下阈值必须严格小于上阈值。")

    # 窗口不足时保留空标签；仅对有效 Z-score 进行状态分类。
    labels = pd.DataFrame(
        pd.NA,
        index=zscores.index,
        columns=zscores.columns,
        dtype="string",
    )
    labels = labels.mask(zscores.notna(), "Neutral")
    labels = labels.mask(zscores > upper_threshold, "Expansion")
    labels = labels.mask(zscores < lower_threshold, "Contraction")
    labels.index.name = "Date"
    labels.columns.name = None
    return labels


def save_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """先写入临时文件再替换目标文件，避免留下不完整 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(
        temporary_path,
        index=True,
        index_label="Date",
        date_format="%Y-%m-%d",
    )
    temporary_path.replace(output_path)


def format_latest_value(value: object) -> str:
    """把最新 Z-score 格式化为便于阅读的文本。"""
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


def format_latest_label(value: object) -> str:
    """把空标签格式化为窗口不足提示。"""
    if pd.isna(value):
        return "数据不足"
    return str(value)


def leading_all_nan_rows(dataframe: pd.DataFrame) -> int:
    """统计首个有效 Z-score 之前，全列均为空的行数。"""
    non_empty_positions = dataframe.notna().any(axis=1).to_numpy().nonzero()[0]
    if len(non_empty_positions) == 0:
        return len(dataframe)
    return int(non_empty_positions[0])


def parse_arguments() -> argparse.Namespace:
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
        help=f"结果 CSV 保存目录（默认：{SCRIPT_DIRECTORY}）",
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
    except (FileNotFoundError, OSError, DataValidationError, pd.errors.ParserError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    latest_date = zscores.index[-1]
    latest_zscores = zscores.iloc[-1]
    latest_labels = regime_labels.iloc[-1]

    print(
        f"阈值设置：Expansion > {arguments.upper_threshold:g}；"
        f"Contraction < {arguments.lower_threshold:g}；其余为 Neutral。"
    )
    print(
        f"前 {leading_all_nan_rows(zscores)} 行因原始窗口和 "
        f"{REFERENCE_WINDOW} 日参考窗口不足产生 NaN，属正常情况。"
    )
    print(f"regime_zscore.csv：{zscores.shape[0]} 行 × {zscores.shape[1]} 列")
    print(
        f"regime_labels.csv：{regime_labels.shape[0]} 行 × "
        f"{regime_labels.shape[1]} 列"
    )
    print(f"最新一期：{latest_date.strftime('%Y-%m-%d')}")
    for ticker in ETF_TICKERS:
        print(
            f"  {ticker}: Z-score={format_latest_value(latest_zscores[ticker])}, "
            f"状态={format_latest_label(latest_labels[ticker])}"
        )
    print(f"结果已保存至：{output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
