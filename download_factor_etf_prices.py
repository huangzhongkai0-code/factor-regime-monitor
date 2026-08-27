"""下载并缓存因子 ETF 的历史日线收盘价。

默认行为：
1. 若 CSV 已存在，则直接读取缓存，避免重复下载。
2. 若缓存不存在，则从 Yahoo Finance 下载数据并保存为 CSV。
3. 输出按自然日重建索引，周末和休市日使用前一交易日收盘价填充。

示例：
    python download_factor_etf_prices.py
    python download_factor_etf_prices.py --refresh
    python download_factor_etf_prices.py --adjusted --output my_prices.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    import pandas as pd
    import yfinance as yf
except ImportError as exc:  # 给出比默认 traceback 更清楚的提示
    missing_package = getattr(exc, "name", "pandas/yfinance")
    raise SystemExit(
        f"缺少依赖 {missing_package!r}。请先运行：pip install pandas yfinance"
    ) from exc


TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
START_DATE = pd.Timestamp("2015-01-01")
# 为起始日是休市日的情况预留前值；最终输出仍从 START_DATE 开始。
DOWNLOAD_START_DATE = START_DATE - pd.Timedelta(days=14)
DEFAULT_OUTPUT = Path(__file__).with_name("factor_etf_close_prices.csv")
DEFAULT_ADJUSTED_OUTPUT = Path(__file__).with_name(
    "factor_etf_adjusted_close_prices.csv"
)


class PriceDownloadError(RuntimeError):
    """行情下载或校验失败。"""


def extract_price_field(
    raw: pd.DataFrame, field: str, tickers: Sequence[str]
) -> pd.DataFrame:
    """兼容 yfinance 的单层/多层列索引，提取指定价格字段。"""
    if raw is None or raw.empty:
        raise PriceDownloadError("Yahoo Finance 未返回任何数据。")

    if isinstance(raw.columns, pd.MultiIndex):
        # 不假定 MultiIndex 一定是 (Price, Ticker)，自动定位价格字段所在层。
        matching_levels = [
            level
            for level in range(raw.columns.nlevels)
            if field in raw.columns.get_level_values(level)
        ]
        if not matching_levels:
            available = sorted(
                {str(value) for value in raw.columns.get_level_values(0)}
            )
            raise PriceDownloadError(
                f"返回数据中没有 {field!r} 字段；可用字段示例：{available[:10]}"
            )

        prices = raw.xs(field, axis=1, level=matching_levels[0], drop_level=True)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        if isinstance(prices.columns, pd.MultiIndex):
            # 正常下载仅余 ticker 一层；额外层出现时将其压平。
            prices.columns = [
                next((str(v) for v in item if str(v) in tickers), str(item[-1]))
                for item in prices.columns.to_flat_index()
            ]
        else:
            prices.columns = prices.columns.astype(str)
    else:
        if field not in raw.columns:
            raise PriceDownloadError(f"返回数据中没有 {field!r} 字段。")
        # 单 ticker 下载通常会走到这个分支。
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]

    # 固定列顺序并检查无效/无数据 ticker。reindex 也能补出完全缺失的列。
    prices = prices.reindex(columns=list(tickers)).apply(pd.to_numeric, errors="coerce")
    prices.columns.name = None
    unavailable = [ticker for ticker in tickers if prices[ticker].notna().sum() == 0]
    if unavailable:
        raise PriceDownloadError(
            "以下 ticker 无有效价格数据（代码可能无效，或服务暂时不可用）："
            + ", ".join(unavailable)
        )

    return prices


def clean_prices(prices: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """去重、按自然日重建索引，并以前值填充休市日和零星缺失值。"""
    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce", utc=True).tz_localize(None)
    cleaned = cleaned.loc[~cleaned.index.isna()]
    cleaned.index = cleaned.index.normalize()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")].sort_index()

    calendar_days = pd.date_range(START_DATE, end_date, freq="D", name="Date")
    # 先保留起始日前的价格用于填充 2015-01-01，再裁剪到要求区间。
    expanded_index = cleaned.index.union(calendar_days)
    cleaned = cleaned.reindex(expanded_index).sort_index().ffill().reindex(calendar_days)
    cleaned = cleaned.reindex(columns=TICKERS)
    cleaned.columns.name = None

    missing_counts = cleaned.isna().sum()
    if missing_counts.any():
        details = ", ".join(
            f"{ticker}={int(count)}"
            for ticker, count in missing_counts.items()
            if count > 0
        )
        raise PriceDownloadError(f"前向填充后仍存在缺失值：{details}")

    return cleaned.astype(float)


def download_prices(adjusted: bool, retries: int = 3) -> pd.DataFrame:
    """下载数据；对网络波动、空响应和无效 ticker 做有限次数重试。"""
    end_date = pd.Timestamp.today().normalize()
    # yfinance 的 end 参数是排他的，因此传入明天以覆盖今天。
    download_end = end_date + pd.Timedelta(days=1)
    field = "Adj Close" if adjusted else "Close"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers=TICKERS,
                start=DOWNLOAD_START_DATE.strftime("%Y-%m-%d"),
                end=download_end.strftime("%Y-%m-%d"),
                interval="1d",
                group_by="column",
                auto_adjust=False,
                actions=False,
                threads=True,
                progress=False,
                timeout=20,
                multi_level_index=True,
            )
            prices = extract_price_field(raw, field, TICKERS)
            return clean_prices(prices, end_date)
        # yfinance 的不同网络后端会抛出不同异常类型，因此在下载边界统一重试。
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"第 {attempt}/{retries} 次下载失败：{exc}；"
                    f"{wait_seconds} 秒后重试……",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)

    raise PriceDownloadError(f"下载在 {retries} 次尝试后仍失败：{last_error}")


def load_cache(path: Path) -> pd.DataFrame:
    """读取并校验本地 CSV 缓存。"""
    cached = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    missing_columns = [ticker for ticker in TICKERS if ticker not in cached.columns]
    if missing_columns:
        raise ValueError("缓存缺少列：" + ", ".join(missing_columns))

    cached = cached.reindex(columns=TICKERS).apply(pd.to_numeric, errors="coerce")
    cached.index = pd.to_datetime(cached.index, errors="coerce")
    cached = cached.loc[~cached.index.isna()].sort_index()
    cached.index.name = "Date"
    if cached.empty or cached.isna().any().any():
        raise ValueError("缓存为空或含有无法解析的缺失值。")
    return cached


def save_cache(prices: pd.DataFrame, path: Path) -> None:
    """先写临时文件再替换，降低写入中断导致缓存损坏的风险。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    prices.to_csv(temporary_path, index=True, index_label="Date", date_format="%Y-%m-%d")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV 保存路径（默认与脚本同目录，并按原始/复权价格分别命名）。",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略已有缓存，重新下载并覆盖 CSV。",
    )
    parser.add_argument(
        "--adjusted",
        action="store_true",
        help="保存复权收盘价 Adj Close；默认保存原始 Close。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_output = DEFAULT_ADJUSTED_OUTPUT if args.adjusted else DEFAULT_OUTPUT
    output_path = (args.output or default_output).expanduser().resolve()

    if output_path.exists() and not args.refresh:
        try:
            prices = load_cache(output_path)
            print(f"已读取本地缓存：{output_path}")
            print(f"数据范围：{prices.index.min().date()} 至 {prices.index.max().date()}")
            print(f"数据形状：{prices.shape[0]} 行 × {prices.shape[1]} 列")
            return 0
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"本地缓存不可用（{exc}），将重新下载。", file=sys.stderr)

    try:
        prices = download_prices(adjusted=args.adjusted)
        save_cache(prices, output_path)
    except (PriceDownloadError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    price_type = "复权收盘价" if args.adjusted else "原始收盘价"
    print(f"{price_type}已保存：{output_path}")
    print(f"数据范围：{prices.index.min().date()} 至 {prices.index.max().date()}")
    print(f"数据形状：{prices.shape[0]} 行 × {prices.shape[1]} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
