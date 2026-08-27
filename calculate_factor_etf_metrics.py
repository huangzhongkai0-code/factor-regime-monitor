"""计算因子 ETF 的日收益率、滚动年化收益率和滚动年化波动率。

输入文件应以 Date 为索引，并包含以下价格列：
SPY、IVE、IVW、MTUM、USMV、IWM。

滚动指标使用 63 个交易日窗口：
- 年化收益率 = 63 日复合增长因子 ** (252 / 63) - 1
- 年化波动率 = 63 日收益率标准差 * sqrt(252)

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


ETF_TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
ROLLING_WINDOW = 63
TRADING_DAYS_PER_YEAR = 252

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "factor_etf_close_prices.csv"
DEFAULT_RETURNS_PATH = SCRIPT_DIRECTORY / "rolling_returns.csv"
DEFAULT_VOLATILITY_PATH = SCRIPT_DIRECTORY / "rolling_volatility.csv"


class DataValidationError(ValueError):
    """输入数据不满足计算要求。"""


def load_and_validate_prices(input_path: Path) -> pd.DataFrame:
    """读取价格 CSV，并校验日期索引、ETF 列和数值内容。"""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    prices = pd.read_csv(input_path, index_col="Date", parse_dates=["Date"])

    missing_tickers = [ticker for ticker in ETF_TICKERS if ticker not in prices.columns]
    if missing_tickers:
        raise DataValidationError("输入文件缺少列：" + ", ".join(missing_tickers))

    # 只保留要求的 ETF，并将无法解析的内容转成 NaN，交由后续逻辑统一处理。
    prices = prices.reindex(columns=ETF_TICKERS).apply(pd.to_numeric, errors="coerce")
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[~prices.index.isna()].sort_index()
    prices.index.name = "Date"
    prices.columns.name = None

    if prices.empty:
        raise DataValidationError("输入文件没有可用数据。")
    if prices.index.duplicated().any():
        duplicated_dates = prices.index[prices.index.duplicated()].strftime("%Y-%m-%d")
        raise DataValidationError("日期索引存在重复值，例如：" + duplicated_dates[0])

    # 零星缺失价格使用前值填充；开头没有前值可用的缺失仍需报错。
    prices = prices.ffill()
    if prices.isna().any().any():
        affected_tickers = prices.columns[prices.isna().any()].tolist()
        raise DataValidationError(
            "前向填充后仍有缺失价格，请检查这些列的起始数据："
            + ", ".join(affected_tickers)
        )
    if (prices <= 0).any().any():
        raise DataValidationError("价格数据包含零或负数，无法可靠计算收益率。")

    return prices.astype(float)


def remove_forward_filled_non_trading_days(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """排除所有 ETF 均与前一日相同的前向填充休市行。

    阶段 1 的价格文件按自然日保存，并将周末/休市日以前值填充。
    这里通过“所有 ETF 同时不变”识别这些行，使窗口按交易日而非自然日计数。
    第一行保留作为下一交易日计算 pct_change 所需的基准价格。
    """
    forward_filled_rows = prices.eq(prices.shift(1)).all(axis=1)
    trading_prices = prices.loc[~forward_filled_rows].copy()
    return trading_prices, int(forward_filled_rows.sum())


def calculate_rolling_metrics(
    trading_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算日收益率及两个 63 交易日滚动年化指标。"""
    # 显式关闭 pct_change 的隐式填充，避免未来 pandas 版本改变计算结果。
    daily_returns = trading_prices.pct_change(fill_method=None).dropna(how="all")

    if len(daily_returns) < ROLLING_WINDOW:
        raise DataValidationError(
            f"有效日收益率仅 {len(daily_returns)} 行，少于 {ROLLING_WINDOW} 日窗口。"
        )

    rolling_window = daily_returns.rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    )

    # 先复合 63 个日收益率，再按一年 252 个交易日进行几何年化。
    rolling_growth_factor = (1.0 + daily_returns).rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).apply(lambda values: values.prod(), raw=True)
    rolling_annualized_returns = (
        rolling_growth_factor.pow(TRADING_DAYS_PER_YEAR / ROLLING_WINDOW) - 1.0
    )

    # pandas 的 std 默认使用样本标准差（ddof=1），再乘 sqrt(252) 年化。
    rolling_annualized_volatility = rolling_window.std(ddof=1) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    for dataframe in (
        daily_returns,
        rolling_annualized_returns,
        rolling_annualized_volatility,
    ):
        dataframe.index.name = "Date"
        dataframe.columns.name = None

    return daily_returns, rolling_annualized_returns, rolling_annualized_volatility


def save_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """先写临时文件再替换目标文件，避免写入中断留下不完整 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(
        temporary_path,
        index=True,
        index_label="Date",
        date_format="%Y-%m-%d",
    )
    temporary_path.replace(output_path)


def first_non_empty_date(dataframe: pd.DataFrame) -> str:
    """返回至少有一个非空指标值的首个日期。"""
    non_empty_rows = dataframe.notna().any(axis=1)
    if not non_empty_rows.any():
        return "无"
    return dataframe.index[non_empty_rows][0].strftime("%Y-%m-%d")


def leading_all_nan_rows(dataframe: pd.DataFrame) -> int:
    """统计首个非空行之前，全列均为 NaN 的行数。"""
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
    arguments = parse_arguments()
    input_path = arguments.input.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    returns_path = output_directory / DEFAULT_RETURNS_PATH.name
    volatility_path = output_directory / DEFAULT_VOLATILITY_PATH.name

    try:
        prices = load_and_validate_prices(input_path)

        # 可选地去掉可能仍是盘中价格的最后一行。
        if arguments.exclude_last_row:
            if len(prices) <= 1:
                raise DataValidationError("数据不足，不能排除最后一行。")
            excluded_date = prices.index[-1].strftime("%Y-%m-%d")
            prices = prices.iloc[:-1]
            print(f"已排除最后一行：{excluded_date}")

        trading_prices, excluded_non_trading_count = (
            remove_forward_filled_non_trading_days(prices)
        )
        daily_returns, rolling_returns, rolling_volatility = (
            calculate_rolling_metrics(trading_prices)
        )

        save_csv_atomically(rolling_returns, returns_path)
        save_csv_atomically(rolling_volatility, volatility_path)
    except (FileNotFoundError, OSError, DataValidationError, pd.errors.ParserError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    leading_nan_count = leading_all_nan_rows(rolling_returns)
    print(f"已识别并排除 {excluded_non_trading_count} 行前向填充的非交易日数据。")
    print(f"日收益率已计算：{daily_returns.shape[0]} 行 × {daily_returns.shape[1]} 列。")
    print(
        f"前 {leading_nan_count} 行因窗口不足产生的 NaN 属正常，"
        "不会被当作数据错误。"
    )
    print(
        f"rolling_returns.csv：{rolling_returns.shape[0]} 行 × "
        f"{rolling_returns.shape[1]} 列；非空数据起始日期："
        f"{first_non_empty_date(rolling_returns)}"
    )
    print(
        f"rolling_volatility.csv：{rolling_volatility.shape[0]} 行 × "
        f"{rolling_volatility.shape[1]} 列；非空数据起始日期："
        f"{first_non_empty_date(rolling_volatility)}"
    )
    print(f"结果已保存至：{output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
