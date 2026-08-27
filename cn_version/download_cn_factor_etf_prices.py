"""下载并缓存中国 A 股场内 ETF 的历史日线收盘价。

默认优先使用 AKShare 的东方财富接口 ``fund_etf_hist_em``；若单只 ETF
请求失败，则自动尝试新浪接口 ``fund_etf_hist_sina``。输出按自然日重建
索引，并用前一交易日收盘价填充周末和休市日。

示例：
    python download_cn_factor_etf_prices.py
    python download_cn_factor_etf_prices.py --refresh
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import akshare as ak
    import pandas as pd
except ImportError as exc:
    missing_package = getattr(exc, "name", "akshare/pandas")
    raise SystemExit(
        f"缺少依赖 {missing_package!r}。请先运行：python -m pip install akshare pandas"
    ) from exc


# 代码顺序同时决定最终 CSV 的列顺序。
ETF_NAMES = {
    "510300": "沪深300ETF（大盘）",
    "510880": "红利ETF（价值/红利）",
    "159915": "创业板ETF（成长）",
    "512100": "中证1000ETF（小盘）",
    "159928": "消费ETF（防御/消费）",
    "588000": "科创50ETF（高弹性成长）",
    "512010": "医药ETF（医药生物）",
    "512480": "半导体ETF（硬科技）",
    "512660": "军工ETF（主题/政策敏感）",
    "512880": "证券ETF（金融/市场情绪）",
    "515030": "新能源车ETF（高景气赛道）",
    "512800": "银行ETF（极致价值/防御）",
}

START_DATE = pd.Timestamp("2015-01-01")
# 起始日前多下载一段数据，使 2015-01-01 恰逢休市时仍可前向填充。
DOWNLOAD_LOOKBACK_DAYS = 14
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIRECTORY / "cn_factor_etf_close_prices.csv"


class ETFDownloadError(RuntimeError):
    """单只 ETF 下载或字段校验失败。"""


class CacheValidationError(ValueError):
    """本地缓存结构不符合预期。"""


@dataclass(frozen=True)
class DownloadResult:
    """保存单只 ETF 的清洗结果和打印所需元数据。"""

    code: str
    name: str
    close_prices: pd.Series
    source: str
    trading_rows: int
    actual_start: pd.Timestamp
    actual_end: pd.Timestamp


def compact_error_message(error: Exception, max_length: int = 240) -> str:
    """把异常压缩成适合终端显示的一行文字。"""

    message = " ".join(str(error).split()) or error.__class__.__name__
    if len(message) > max_length:
        message = message[: max_length - 3] + "..."
    return f"{error.__class__.__name__}: {message}"


def exchange_prefixed_symbol(code: str) -> str:
    """把六位 ETF 代码转换为新浪接口使用的 sh/sz 前缀格式。"""

    if len(code) != 6 or not code.isdigit():
        raise ETFDownloadError(f"ETF 代码格式无效：{code!r}")
    if code.startswith(("5", "6")):
        return f"sh{code}"
    if code.startswith(("1", "0", "3")):
        return f"sz{code}"
    raise ETFDownloadError(f"无法判断 ETF {code} 所属交易所。")


def normalize_close_prices(
    raw_data: pd.DataFrame,
    code: str,
    source_key: str,
    download_start: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """兼容东财/新浪字段名，并清洗为以交易日期为索引的收盘价序列。"""

    if raw_data is None or raw_data.empty:
        raise ETFDownloadError(
            f"{code} 未返回任何数据，可能是代码无效、尚未上市或数据源暂不可用。"
        )

    if source_key == "em":
        date_column, close_column = "日期", "收盘"
    elif source_key == "sina":
        date_column, close_column = "date", "close"
    else:
        raise ETFDownloadError(f"未知数据源：{source_key}")

    missing_columns = [
        column
        for column in (date_column, close_column)
        if column not in raw_data.columns
    ]
    if missing_columns:
        available_columns = [str(column) for column in raw_data.columns]
        raise ETFDownloadError(
            "AKShare 接口返回字段可能已变化；"
            f"缺少 {missing_columns}，实际字段为 {available_columns}。"
        )

    cleaned_data = raw_data[[date_column, close_column]].copy()
    cleaned_data[date_column] = pd.to_datetime(
        cleaned_data[date_column], errors="coerce"
    )
    cleaned_data[close_column] = pd.to_numeric(
        cleaned_data[close_column], errors="coerce"
    )

    # 日期无效、价格缺失或非正数均不属于可用收盘价。
    cleaned_data = cleaned_data.dropna(subset=[date_column, close_column])
    cleaned_data = cleaned_data.loc[cleaned_data[close_column] > 0]
    cleaned_data = cleaned_data.drop_duplicates(subset=[date_column], keep="last")
    cleaned_data = cleaned_data.sort_values(date_column)
    cleaned_data = cleaned_data.loc[
        cleaned_data[date_column].between(download_start, end_date)
    ]

    if cleaned_data.empty:
        raise ETFDownloadError(
            f"{code} 在 {download_start:%Y-%m-%d} 至 {end_date:%Y-%m-%d} "
            "之间没有有效收盘价。"
        )

    close_prices = cleaned_data.set_index(date_column)[close_column].astype(float)
    close_prices.index = pd.DatetimeIndex(close_prices.index).normalize()
    close_prices.index.name = "Date"
    close_prices.name = code
    return close_prices


def request_eastmoney(
    code: str, download_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """通过 AKShare 东方财富 ETF 日线接口请求数据。"""

    interface = getattr(ak, "fund_etf_hist_em", None)
    if not callable(interface):
        raise ETFDownloadError(
            "当前 AKShare 中不存在 fund_etf_hist_em，接口名称可能已变化。"
        )
    return interface(
        symbol=code,
        period="daily",
        start_date=download_start.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="",
    )


def request_sina(
    code: str, download_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """通过 AKShare 新浪 ETF 日线接口请求数据；日期筛选在清洗阶段完成。"""

    del download_start, end_date  # 新浪接口一次返回全部历史数据。
    interface = getattr(ak, "fund_etf_hist_sina", None)
    if not callable(interface):
        raise ETFDownloadError(
            "当前 AKShare 中不存在 fund_etf_hist_sina，备用接口名称可能已变化。"
        )
    return interface(symbol=exchange_prefixed_symbol(code))


SOURCE_CONFIG: dict[str, tuple[str, Callable[..., pd.DataFrame]]] = {
    "em": ("东方财富 fund_etf_hist_em", request_eastmoney),
    "sina": ("新浪 fund_etf_hist_sina", request_sina),
}


def request_with_retries(
    code: str,
    source_key: str,
    download_start: pd.Timestamp,
    end_date: pd.Timestamp,
    retries: int,
    retry_delay: float,
) -> pd.Series:
    """请求并清洗单一数据源；网络或数据异常时按指定次数重试。"""

    source_name, request_function = SOURCE_CONFIG[source_key]
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            raw_data = request_function(code, download_start, end_date)
            return normalize_close_prices(
                raw_data=raw_data,
                code=code,
                source_key=source_key,
                download_start=download_start,
                end_date=end_date,
            )
        except Exception as error:  # AKShare 底层可能抛出多种网络异常。
            last_error = error
            print(
                f"    [重试] {source_name} 第 {attempt}/{retries} 次失败："
                f"{compact_error_message(error)}"
            )
            if attempt < retries:
                time.sleep(retry_delay * attempt)

    assert last_error is not None
    raise ETFDownloadError(
        f"{source_name} 连续 {retries} 次请求失败："
        f"{compact_error_message(last_error)}"
    ) from last_error


def download_one_etf(
    code: str,
    name: str,
    download_start: pd.Timestamp,
    end_date: pd.Timestamp,
    source_mode: str,
    retries: int,
    retry_delay: float,
) -> DownloadResult:
    """下载一只 ETF；auto 模式下东财失败后自动尝试新浪。"""

    source_order = ["em", "sina"] if source_mode == "auto" else [source_mode]
    source_errors: list[str] = []

    for source_key in source_order:
        source_name = SOURCE_CONFIG[source_key][0]
        try:
            close_prices = request_with_retries(
                code=code,
                source_key=source_key,
                download_start=download_start,
                end_date=end_date,
                retries=retries,
                retry_delay=retry_delay,
            )
            analysis_period = close_prices.loc[
                (close_prices.index >= START_DATE)
                & (close_prices.index <= end_date)
            ]
            if analysis_period.empty:
                raise ETFDownloadError(
                    f"{code} 从 2015 年至今没有有效交易数据。"
                )
            return DownloadResult(
                code=code,
                name=name,
                close_prices=close_prices,
                source=source_name,
                trading_rows=int(analysis_period.shape[0]),
                actual_start=analysis_period.index.min(),
                actual_end=analysis_period.index.max(),
            )
        except Exception as error:
            source_errors.append(f"{source_name}：{compact_error_message(error)}")
            if source_key != source_order[-1]:
                print(f"    [提示] {source_name}不可用，自动切换备用数据源。")

    raise ETFDownloadError("；".join(source_errors))


def build_calendar_dataframe(
    results: dict[str, DownloadResult], end_date: pd.Timestamp
) -> pd.DataFrame:
    """合并各 ETF，并按自然日重建索引后进行前向填充。"""

    download_start = START_DATE - pd.Timedelta(days=DOWNLOAD_LOOKBACK_DAYS)
    extended_calendar = pd.date_range(download_start, end_date, freq="D")
    output_calendar = pd.date_range(START_DATE, end_date, freq="D", name="Date")

    series_by_code = {
        code: result.close_prices for code, result in results.items()
    }
    combined_prices = pd.concat(series_by_code, axis=1) if series_by_code else pd.DataFrame()
    combined_prices = combined_prices.reindex(extended_calendar).ffill()
    combined_prices = combined_prices.reindex(output_calendar)

    # 即使个别代码失败，也保留完整的 12 列结构；失败代码对应列为空。
    combined_prices = combined_prices.reindex(columns=list(ETF_NAMES))
    combined_prices.columns.name = None
    combined_prices.index.name = "Date"
    return combined_prices


def save_csv_atomically(prices: pd.DataFrame, output_path: Path) -> None:
    """先写临时文件再替换正式文件，避免中途失败留下半个 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        prices.to_csv(
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


def load_cached_prices(output_path: Path) -> pd.DataFrame:
    """读取缓存并严格校验日期索引、列顺序和数值类型。"""

    try:
        prices = pd.read_csv(
            output_path,
            index_col="Date",
            parse_dates=["Date"],
            encoding="utf-8-sig",
        )
    except Exception as error:
        raise CacheValidationError(
            f"无法读取缓存文件：{compact_error_message(error)}"
        ) from error

    missing_codes = [code for code in ETF_NAMES if code not in prices.columns]
    if missing_codes:
        raise CacheValidationError("缓存缺少 ETF 列：" + ", ".join(missing_codes))

    prices = prices.reindex(columns=list(ETF_NAMES))
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.isna().any():
        raise CacheValidationError("缓存的 Date 索引包含无法识别的日期。")
    if prices.index.has_duplicates:
        raise CacheValidationError("缓存的 Date 索引包含重复日期。")
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    if prices.empty or not prices.notna().any().any():
        raise CacheValidationError("缓存为空或没有任何有效价格。")
    expected_calendar = pd.date_range(
        prices.index.min(), prices.index.max(), freq="D", name="Date"
    )
    if not prices.index.equals(expected_calendar):
        raise CacheValidationError("缓存未按连续自然日建立 Date 索引。")

    prices.index.name = "Date"
    return prices


def print_cache_summary(prices: pd.DataFrame, output_path: Path) -> None:
    """缓存模式下打印每列非空形状和有效数据起始日期。"""

    print(f"[缓存] 已读取：{output_path}")
    print(f"[缓存] 整体形状：{prices.shape[0]} 行 × {prices.shape[1]} 列")
    for code, name in ETF_NAMES.items():
        valid_prices = prices[code].dropna()
        if valid_prices.empty:
            print(f"  [无数据] {code} {name}：缓存列为空")
        else:
            print(
                f"  [缓存] {code} {name}：非空形状 ({len(valid_prices)}, 1)，"
                f"有效起始日 {valid_prices.index.min():%Y-%m-%d}，"
                f"截至 {valid_prices.index.max():%Y-%m-%d}"
            )
    print("如需忽略缓存并重新下载，请加参数：--refresh")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略已有 CSV 缓存，强制重新下载。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV 输出路径，默认：{DEFAULT_OUTPUT_PATH.name}",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "em", "sina"),
        default="auto",
        help="数据源：auto=东财失败后转新浪；em=仅东财；sina=仅新浪。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="每个数据源的最大尝试次数，默认 2。",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.5,
        help="首次重试等待秒数，后续按次数递增，默认 1.5。",
    )
    arguments = parser.parse_args()
    if arguments.retries < 1:
        parser.error("--retries 必须大于或等于 1。")
    if arguments.retry_delay < 0:
        parser.error("--retry-delay 不能为负数。")
    return arguments


def main() -> int:
    """程序入口：优先读取缓存，否则逐只下载、合并并保存。"""

    arguments = parse_arguments()
    output_path = arguments.output.expanduser().resolve()

    if output_path.exists() and not arguments.refresh:
        try:
            cached_prices = load_cached_prices(output_path)
            print_cache_summary(cached_prices, output_path)
            return 0
        except CacheValidationError as error:
            print(f"[错误] 本地缓存校验失败：{error}", file=sys.stderr)
            print("请检查文件，或使用 --refresh 重新下载。", file=sys.stderr)
            return 1

    end_date = pd.Timestamp.today().normalize()
    download_start = START_DATE - pd.Timedelta(days=DOWNLOAD_LOOKBACK_DAYS)
    successful_results: dict[str, DownloadResult] = {}
    failed_downloads: dict[str, str] = {}

    print(f"AKShare 版本：{getattr(ak, '__version__', '未知')}")
    print(
        f"下载区间：{START_DATE:%Y-%m-%d} 至 {end_date:%Y-%m-%d}；"
        f"模式：{arguments.source}"
    )
    print("未指定复权参数，保存交易所原始日线收盘价。")

    for position, (code, name) in enumerate(ETF_NAMES.items(), start=1):
        print(f"\n[{position}/{len(ETF_NAMES)}] 正在下载 {code} {name}...")
        try:
            result = download_one_etf(
                code=code,
                name=name,
                download_start=download_start,
                end_date=end_date,
                source_mode=arguments.source,
                retries=arguments.retries,
                retry_delay=arguments.retry_delay,
            )
            successful_results[code] = result
            print(
                f"  [成功] 原始交易日数据形状 ({result.trading_rows}, 1)；"
                f"实际起始日 {result.actual_start:%Y-%m-%d}；"
                f"截至 {result.actual_end:%Y-%m-%d}；来源：{result.source}"
            )
        except Exception as error:
            error_message = compact_error_message(error)
            failed_downloads[code] = error_message
            print(f"  [跳过] {code} 下载失败：{error_message}", file=sys.stderr)

        # 对公开数据源保持适度请求间隔，降低触发限流的概率。
        if position < len(ETF_NAMES):
            time.sleep(0.4)

    if not successful_results:
        print("\n[错误] 12 只 ETF 全部下载失败，未生成 CSV。", file=sys.stderr)
        return 1

    prices = build_calendar_dataframe(successful_results, end_date)
    try:
        save_csv_atomically(prices, output_path)
    except Exception as error:
        print(
            f"\n[错误] CSV 保存失败：{compact_error_message(error)}",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 68)
    print(f"已保存 CSV：{output_path}")
    print(f"最终 DataFrame 形状：{prices.shape[0]} 行 × {prices.shape[1]} 列")
    print(
        f"自然日范围：{prices.index.min():%Y-%m-%d} 至 "
        f"{prices.index.max():%Y-%m-%d}"
    )
    print(f"成功：{len(successful_results)} 只；失败：{len(failed_downloads)} 只")

    if failed_downloads:
        print("\n[警告] 以下 ETF 下载失败，CSV 中保留对应空列：")
        for code, error_message in failed_downloads.items():
            print(f"  - {code} {ETF_NAMES[code]}：{error_message}")
        print("其他 ETF 已正常保存；可稍后使用 --refresh 重试。")
    else:
        print("全部 12 只 ETF 下载成功。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
