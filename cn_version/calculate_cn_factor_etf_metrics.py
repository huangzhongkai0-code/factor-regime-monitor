"""计算中国 A 股因子 ETF 的日收益率及 3 个月滚动年化指标。

输入文件应以 Date 为索引，并包含 12 只 ETF 的日线收盘价。价格文件按
自然日保存，因此计算前会删除“所有已有价格的 ETF 均与前一日相同”的
前向填充行，使 63 日窗口按真实交易日计数。

计算口径：
- 日收益率：pct_change(fill_method=None)
- 63 日滚动年化收益率：63 日复合增长因子 ** (252 / 63) - 1
- 63 日滚动年化波动率：日收益率样本标准差 * sqrt(252)

CSV 中的收益率和波动率均以小数表示，例如 0.12 代表 12%。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("缺少 pandas。请先运行：python -m pip install pandas") from exc


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

ROLLING_WINDOW = 63
TRADING_DAYS_PER_YEAR = 252

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "cn_factor_etf_close_prices.csv"
DEFAULT_RETURNS_FILENAME = "cn_rolling_returns.csv"
DEFAULT_VOLATILITY_FILENAME = "cn_rolling_volatility.csv"


class DataValidationError(ValueError):
    """输入数据不满足计算要求。"""


def load_and_validate_prices(input_path: Path) -> pd.DataFrame:
    """读取价格 CSV，并校验日期索引、ETF 列和价格数值。"""

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    try:
        raw_prices = pd.read_csv(
            input_path,
            index_col="Date",
            parse_dates=["Date"],
            encoding="utf-8-sig",
        )
    except ValueError as error:
        raise DataValidationError(
            "无法按 Date 索引读取输入 CSV，请检查表头和日期列。"
        ) from error

    missing_codes = [code for code in ETF_CODES if code not in raw_prices.columns]
    if missing_codes:
        raise DataValidationError("输入文件缺少 ETF 列：" + ", ".join(missing_codes))

    prices = raw_prices.reindex(columns=ETF_CODES).copy()
    parsed_index = pd.to_datetime(prices.index, errors="coerce")
    if parsed_index.isna().any():
        raise DataValidationError("Date 索引包含无法识别的日期。")
    prices.index = parsed_index
    prices = prices.sort_index()

    if prices.index.has_duplicates:
        duplicated_date = prices.index[prices.index.duplicated()][0]
        raise DataValidationError(
            f"Date 索引存在重复日期，例如：{duplicated_date:%Y-%m-%d}"
        )

    # 上市前空值是正常现象；仅把非空但无法转成数字的内容视为数据错误。
    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    invalid_numeric_mask = prices.notna() & numeric_prices.isna()
    if invalid_numeric_mask.any().any():
        row_position, column_position = (
            invalid_numeric_mask.to_numpy().nonzero()[0][0],
            invalid_numeric_mask.to_numpy().nonzero()[1][0],
        )
        invalid_date = prices.index[row_position]
        invalid_code = prices.columns[column_position]
        raise DataValidationError(
            f"{invalid_code} 在 {invalid_date:%Y-%m-%d} 包含无法解析的价格。"
        )

    non_positive_mask = numeric_prices.notna() & (numeric_prices <= 0)
    if non_positive_mask.any().any():
        row_position, column_position = (
            non_positive_mask.to_numpy().nonzero()[0][0],
            non_positive_mask.to_numpy().nonzero()[1][0],
        )
        invalid_date = numeric_prices.index[row_position]
        invalid_code = numeric_prices.columns[column_position]
        raise DataValidationError(
            f"{invalid_code} 在 {invalid_date:%Y-%m-%d} 的价格为零或负数。"
        )

    if numeric_prices.empty or not numeric_prices.notna().any().any():
        raise DataValidationError("输入文件为空或没有任何有效价格。")

    numeric_prices.index.name = "Date"
    numeric_prices.columns.name = None
    return numeric_prices.astype(float)


def remove_forward_filled_non_trading_days(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """删除由前向填充形成的全市场非交易日行。

    由于各 ETF 上市时间不同，上市前的 NaN 不参与“是否与前一日相同”的
    比较。只要至少有一只 ETF 在相邻两日都有价格，且所有可比较 ETF 都
    没有变化，就把该行识别为非交易日。若某只 ETF 当日首次出现价格，
    则强制保留该行，避免误删上市日。
    """

    previous_prices = prices.shift(1)
    comparable_mask = prices.notna() & previous_prices.notna()
    availability_changed = prices.notna().ne(previous_prices.notna()).any(axis=1)

    unchanged_or_not_comparable = prices.eq(previous_prices) | ~comparable_mask
    forward_filled_rows = (
        comparable_mask.any(axis=1)
        & unchanged_or_not_comparable.all(axis=1)
        & ~availability_changed
    )

    trading_prices = prices.loc[~forward_filled_rows].copy()
    if len(trading_prices) < 2:
        raise DataValidationError("排除非交易日后数据不足两行，无法计算收益率。")

    return trading_prices, int(forward_filled_rows.sum())


def calculate_rolling_metrics(
    trading_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """逐列独立计算日收益率、滚动年化收益率和年化波动率。"""

    # fill_method=None 可防止 pandas 对上市前或内部缺失价格进行隐式填充。
    daily_returns = trading_prices.pct_change(fill_method=None).dropna(how="all")
    if daily_returns.empty:
        raise DataValidationError("没有可用的日收益率，无法继续计算。")

    # 每列使用自身最近 63 个非缺失日收益率；上市前数据自然保持 NaN。
    rolling_growth_factor = (1.0 + daily_returns).rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).apply(lambda values: values.prod(), raw=True)
    rolling_annualized_returns = (
        rolling_growth_factor.pow(TRADING_DAYS_PER_YEAR / ROLLING_WINDOW) - 1.0
    )

    rolling_annualized_volatility = daily_returns.rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)

    for dataframe in (
        daily_returns,
        rolling_annualized_returns,
        rolling_annualized_volatility,
    ):
        dataframe.index.name = "Date"
        dataframe.columns.name = None

    return daily_returns, rolling_annualized_returns, rolling_annualized_volatility


def save_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """先保存临时文件再替换目标文件，避免中断后留下不完整 CSV。"""

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
    """把首个有效日期转换成终端友好的字符串。"""

    return "无" if date_value is None else date_value.strftime("%Y-%m-%d")


def print_per_ticker_summary(
    trading_prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rolling_returns: pd.DataFrame,
    rolling_volatility: pd.DataFrame,
) -> None:
    """打印每只 ETF 的价格、日收益率和滚动指标有效起始日期。"""

    print("\n每只 ETF 的有效数据起始日期：")
    for code in ETF_CODES:
        price_start = trading_prices[code].first_valid_index()
        daily_return_start = daily_returns[code].first_valid_index()
        rolling_return_start = rolling_returns[code].first_valid_index()
        rolling_volatility_start = rolling_volatility[code].first_valid_index()
        print(
            f"  {code}：价格 {format_date(price_start)}；"
            f"日收益率 {format_date(daily_return_start)}；"
            f"滚动收益率 {format_date(rolling_return_start)}；"
            f"滚动波动率 {format_date(rolling_volatility_start)}"
        )


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"输入价格 CSV（默认：{DEFAULT_INPUT_PATH}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIRECTORY,
        help=f"两个结果 CSV 的保存目录（默认：{SCRIPT_DIRECTORY}）",
    )
    parser.add_argument(
        "--exclude-last-row",
        action="store_true",
        help="计算前删除输入数据最后一行，适用于当天价格尚未收盘的情况。",
    )
    return parser.parse_args()


def main() -> int:
    """程序入口：读取、筛选交易日、计算指标并保存结果。"""

    arguments = parse_arguments()
    input_path = arguments.input.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    returns_path = output_directory / DEFAULT_RETURNS_FILENAME
    volatility_path = output_directory / DEFAULT_VOLATILITY_FILENAME

    try:
        prices = load_and_validate_prices(input_path)

        # 可选排除最后一个自然日，防止盘中价格混入完整日线计算。
        if arguments.exclude_last_row:
            if len(prices) <= 1:
                raise DataValidationError("数据不足，不能排除最后一行。")
            excluded_date = prices.index[-1]
            prices = prices.iloc[:-1].copy()
            print(f"已排除输入数据最后一行：{excluded_date:%Y-%m-%d}")

        trading_prices, excluded_non_trading_count = (
            remove_forward_filled_non_trading_days(prices)
        )
        daily_returns, rolling_returns, rolling_volatility = (
            calculate_rolling_metrics(trading_prices)
        )

        save_csv_atomically(rolling_returns, returns_path)
        save_csv_atomically(rolling_volatility, volatility_path)
    except (
        FileNotFoundError,
        OSError,
        DataValidationError,
        pd.errors.ParserError,
    ) as error:
        print(f"[错误] {error}", file=sys.stderr)
        return 1

    print(f"已识别并排除 {excluded_non_trading_count} 行前向填充的非交易日。")
    print(
        f"真实交易日价格：{trading_prices.shape[0]} 行 × "
        f"{trading_prices.shape[1]} 列"
    )
    print(f"日收益率：{daily_returns.shape[0]} 行 × {daily_returns.shape[1]} 列")
    print_per_ticker_summary(
        trading_prices=trading_prices,
        daily_returns=daily_returns,
        rolling_returns=rolling_returns,
        rolling_volatility=rolling_volatility,
    )
    print(
        f"\n{DEFAULT_RETURNS_FILENAME}："
        f"{rolling_returns.shape[0]} 行 × {rolling_returns.shape[1]} 列"
    )
    print(
        f"{DEFAULT_VOLATILITY_FILENAME}："
        f"{rolling_volatility.shape[0]} 行 × {rolling_volatility.shape[1]} 列"
    )
    print(f"结果已保存至：{output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
