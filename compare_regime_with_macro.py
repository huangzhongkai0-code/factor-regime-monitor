"""将因子状态与 FRED 美国宏观数据按月对齐并进行交叉验证。

环境变量：
    FRED_API_KEY    FRED API 密钥（必需）

默认输入：
    regime_labels.csv
    regime_zscore.csv

默认输出：
    regime_vs_macro.png
    macro_correlation_summary.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

try:
    import matplotlib

    # 使用非交互式后端，确保脚本能在终端或自动化任务中运行。
    matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from fredapi import Fred
    from matplotlib import font_manager
except ImportError as exc:
    missing_package = getattr(exc, "name", "pandas/matplotlib/fredapi")
    raise SystemExit(
        f"缺少依赖 {missing_package!r}。请先运行："
        "python -m pip install pandas matplotlib fredapi"
    ) from exc


ETF_TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
FACTOR_NAMES = {
    "SPY": "大盘",
    "IVE": "价值因子",
    "IVW": "成长因子",
    "MTUM": "动量因子",
    "USMV": "低波动因子",
    "IWM": "小盘股因子",
}

FRED_SERIES = {
    "INDPRO": "工业生产指数",
    "CPIAUCSL": "消费者价格指数",
}
ANALYSIS_START_DATE = pd.Timestamp("2015-01-01")
# CPI 同比需要12个月前值，因此从2014年开始下载宏观数据。
FRED_DOWNLOAD_START_DATE = pd.Timestamp("2014-01-01")
CORRELATION_WINDOW_MONTHS = 12

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Microsoft JhengHei",
    "PingFang SC",
    "Arial Unicode MS",
]

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_LABELS_PATH = SCRIPT_DIRECTORY / "regime_labels.csv"
DEFAULT_ZSCORE_PATH = SCRIPT_DIRECTORY / "regime_zscore.csv"
CHART_FILENAME = "regime_vs_macro.png"
SUMMARY_FILENAME = "macro_correlation_summary.txt"


class MacroAnalysisError(ValueError):
    """输入、FRED 下载或宏观分析失败。"""


def get_fred_api_key() -> str:
    """从环境变量读取 FRED API key，不允许在代码中硬编码。"""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise MacroAnalysisError(
            "未设置环境变量 FRED_API_KEY。Windows PowerShell 可运行：\n"
            "[Environment]::SetEnvironmentVariable("
            "'FRED_API_KEY', '你的FRED API key', 'User')\n"
            "设置后请重新打开终端再运行脚本。"
        )
    if len(api_key) != 32 or not api_key.isalnum():
        raise MacroAnalysisError(
            "环境变量 FRED_API_KEY 的格式不正确，应为32位字母数字字符串。"
        )
    return api_key


def configure_chinese_font() -> str:
    """选择已安装的中文字体，避免 Matplotlib 中文乱码。"""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    raise MacroAnalysisError(
        "未检测到中文字体。请安装微软雅黑、黑体或 Noto Sans CJK SC 后重试。"
    )


def read_csv_with_date_index(file_path: Path) -> pd.DataFrame:
    """读取以 Date 为索引的 CSV，并规范日期顺序。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{file_path}")

    dataframe = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
    dataframe.index = pd.to_datetime(dataframe.index, errors="coerce")
    dataframe = dataframe.loc[~dataframe.index.isna()].sort_index()
    dataframe.index.name = "Date"
    dataframe.columns.name = None

    if dataframe.empty:
        raise MacroAnalysisError(f"输入文件没有可用数据：{file_path}")
    if dataframe.index.duplicated().any():
        duplicated_date = dataframe.index[dataframe.index.duplicated()][0]
        raise MacroAnalysisError(
            f"{file_path.name} 存在重复日期：{duplicated_date:%Y-%m-%d}"
        )
    return dataframe


def load_and_validate_factor_data(
    labels_path: Path,
    zscore_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并校验交易日频率的状态标签和 Z-score。"""
    labels = read_csv_with_date_index(labels_path)
    zscores = read_csv_with_date_index(zscore_path)

    for file_name, dataframe in (
        (labels_path.name, labels),
        (zscore_path.name, zscores),
    ):
        missing_tickers = [
            ticker for ticker in ETF_TICKERS if ticker not in dataframe.columns
        ]
        if missing_tickers:
            raise MacroAnalysisError(
                f"{file_name} 缺少列：" + ", ".join(missing_tickers)
            )

    labels = labels.reindex(columns=ETF_TICKERS).apply(
        lambda column: column.astype("string").str.strip()
    )
    zscores = zscores.reindex(columns=ETF_TICKERS).apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not labels.index.equals(zscores.index):
        raise MacroAnalysisError("regime_labels.csv 与 regime_zscore.csv 日期不一致。")

    allowed_labels = {"Expansion", "Neutral", "Contraction"}
    invalid_label_mask = labels.notna() & ~labels.isin(allowed_labels)
    if invalid_label_mask.any().any():
        row_position, column_position = np.argwhere(invalid_label_mask.to_numpy())[0]
        raise MacroAnalysisError(
            "发现无效状态标签："
            f"{labels.index[row_position]:%Y-%m-%d} "
            f"{labels.columns[column_position]}="
            f"{labels.iloc[row_position, column_position]!r}"
        )

    return labels, zscores.astype(float)


def redact_secret(message: str, api_key: str) -> str:
    """确保异常信息不会意外包含 API key。"""
    return message.replace(api_key, "[REDACTED]")


def download_fred_series(
    fred_client: Fred,
    series_id: str,
    api_key: str,
) -> pd.Series:
    """下载单个 FRED 序列，并把网络/API错误转换成清晰提示。"""
    try:
        series = fred_client.get_series(
            series_id,
            observation_start=FRED_DOWNLOAD_START_DATE,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        safe_message = redact_secret(str(exc), api_key)
        raise MacroAnalysisError(
            f"FRED 序列 {series_id}（{FRED_SERIES[series_id]}）下载失败："
            f"{safe_message or type(exc).__name__}。"
            "请检查网络、API key 和序列代码后重试。"
        ) from exc

    if series is None or series.empty:
        raise MacroAnalysisError(
            f"FRED 序列 {series_id}（{FRED_SERIES[series_id]}）没有返回数据。"
        )

    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[~series.index.isna()]
    series.name = series_id
    if series.empty:
        raise MacroAnalysisError(f"FRED 序列 {series_id} 没有可用数值。")
    return series.astype(float)


def convert_to_month_end_index(dataframe: pd.DataFrame | pd.Series):
    """把 FRED 的月初日期规范为月末日期，便于与因子月度数据合并。"""
    converted = dataframe.copy()
    converted.index = converted.index.to_period("M").to_timestamp("M")
    converted.index.name = "Date"
    return converted


def build_monthly_merged_data(
    daily_labels: pd.DataFrame,
    daily_zscores: pd.DataFrame,
    industrial_production: pd.Series,
    cpi_index: pd.Series,
) -> pd.DataFrame:
    """将日频因子数据和月频宏观数据统一到月末频率。"""
    # 每月取最后一个交易日的状态标签和 Z-score。
    monthly_labels = daily_labels.resample("ME").last()
    monthly_zscores = daily_zscores.resample("ME").last()
    monthly_labels.columns = [f"{ticker}_Regime" for ticker in ETF_TICKERS]
    monthly_zscores.columns = [f"{ticker}_Zscore" for ticker in ETF_TICKERS]

    industrial_production = convert_to_month_end_index(industrial_production)
    cpi_index = convert_to_month_end_index(cpi_index)

    macro_data = pd.concat(
        [industrial_production.rename("INDPRO"), cpi_index.rename("CPIAUCSL")],
        axis=1,
        sort=False,
    ).sort_index()
    # CPI同比增速以百分比表示，例如 3.2 代表同比上涨3.2%。
    macro_data["CPI_YoY"] = macro_data["CPIAUCSL"].pct_change(
        periods=12,
        fill_method=None,
    ) * 100.0

    merged_monthly_data = pd.concat(
        [monthly_zscores, monthly_labels, macro_data],
        axis=1,
        sort=False,
    ).sort_index()

    analysis_end_date = monthly_zscores.index.max()
    merged_monthly_data = merged_monthly_data.loc[
        ANALYSIS_START_DATE:analysis_end_date
    ]
    merged_monthly_data.index.name = "Date"

    required_factor_columns = [f"{ticker}_Zscore" for ticker in ETF_TICKERS]
    if merged_monthly_data[required_factor_columns].dropna(how="all").empty:
        raise MacroAnalysisError("月度因子 Z-score 全部为空，无法继续分析。")
    if merged_monthly_data["CPI_YoY"].dropna().empty:
        raise MacroAnalysisError("CPI同比增速全部为空，无法继续分析。")

    return merged_monthly_data


def calculate_rolling_correlations(
    monthly_data: pd.DataFrame,
) -> pd.DataFrame:
    """计算各因子月度 Z-score 与 CPI 同比的12个月滚动相关系数。"""
    rolling_correlations = pd.DataFrame(index=monthly_data.index)
    for ticker in ETF_TICKERS:
        rolling_correlations[ticker] = monthly_data[f"{ticker}_Zscore"].rolling(
            window=CORRELATION_WINDOW_MONTHS,
            min_periods=CORRELATION_WINDOW_MONTHS,
        ).corr(monthly_data["CPI_YoY"])

    rolling_correlations.index.name = "Date"
    rolling_correlations.columns.name = None
    if rolling_correlations.dropna(how="all").empty:
        raise MacroAnalysisError(
            "没有形成完整的12个月相关性窗口，请检查月度数据长度和缺失值。"
        )
    return rolling_correlations


def create_comparison_chart(
    monthly_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制共享时间轴的上下双面板图。"""
    figure, (factor_axis, cpi_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 8.6),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.10},
    )

    factor_axis.plot(
        monthly_data.index,
        monthly_data["SPY_Zscore"],
        color="#315A9B",
        linewidth=1.8,
        label="SPY（大盘）Z-score",
    )
    factor_axis.plot(
        monthly_data.index,
        monthly_data["USMV_Zscore"],
        color="#2E8B67",
        linewidth=1.8,
        label="USMV（低波动）Z-score",
    )
    factor_axis.axhline(0, color="#6B7280", linewidth=0.8)
    factor_axis.axhline(1, color="#2E8B67", linewidth=0.9, linestyle="--", alpha=0.7)
    factor_axis.axhline(-1, color="#C85A63", linewidth=0.9, linestyle="--", alpha=0.7)
    factor_axis.set_ylabel("月度 Z-score")
    factor_axis.set_title("SPY 与 USMV 月度状态强度")
    factor_axis.legend(loc="upper left", frameon=False, ncol=2)

    cpi_axis.plot(
        monthly_data.index,
        monthly_data["CPI_YoY"],
        color="#C77724",
        linewidth=2.0,
        label="CPI同比增速",
    )
    cpi_axis.axhline(0, color="#6B7280", linewidth=0.8)
    cpi_axis.set_ylabel("CPI同比（%）")
    cpi_axis.set_xlabel("月份")
    cpi_axis.set_title("美国 CPI 同比增速")
    cpi_axis.legend(loc="upper left", frameon=False)

    for axis in (factor_axis, cpi_axis):
        axis.grid(axis="y", color="#D7DCE3", linewidth=0.7, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#8B95A5")
        axis.spines["bottom"].set_color("#8B95A5")

    cpi_axis.xaxis.set_major_locator(mdates.YearLocator())
    cpi_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    cpi_axis.set_xlim(monthly_data.index.min(), monthly_data.index.max())

    figure.suptitle(
        "因子状态与美国通胀走势交叉验证（2015年至今）",
        y=0.97,
        fontweight="normal",
    )
    figure.text(
        0.99,
        0.012,
        "数据源：FRED（INDPRO、CPIAUCSL）；因子数据按月取最后一个交易日",
        ha="right",
        va="bottom",
        color="#5F6875",
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.91)
    figure.patch.set_facecolor("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    figure.savefig(
        temporary_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    temporary_path.replace(output_path)


def format_correlation_item(ticker: str, correlation: float) -> str:
    """格式化相关系数及其方向。"""
    direction = "正相关" if correlation >= 0 else "负相关"
    return (
        f"{ticker}（{FACTOR_NAMES[ticker]}，r={correlation:.2f}，{direction}）"
    )


def generate_correlation_summary(
    rolling_correlations: pd.DataFrame,
    monthly_data: pd.DataFrame,
) -> str:
    """生成保守克制的中文相关性总结。"""
    complete_correlation_rows = rolling_correlations.dropna(how="any")
    if complete_correlation_rows.empty:
        raise MacroAnalysisError("各因子没有共同有效的12个月滚动相关系数。")

    latest_date = complete_correlation_rows.index[-1]
    latest_correlations = complete_correlation_rows.iloc[-1]
    absolute_ranking = latest_correlations.abs().sort_values(ascending=False)
    stronger_tickers = absolute_ranking.index[:2].tolist()
    weaker_tickers = absolute_ranking.index[-2:].tolist()

    historical_median_absolute = rolling_correlations.abs().median().sort_values(
        ascending=False
    )
    historically_higher_tickers = historical_median_absolute.index[:2].tolist()

    strongest_absolute_value = float(absolute_ranking.iloc[0])
    if strongest_absolute_value >= 0.6:
        overall_description = "最新窗口中部分因子与通胀观察到较明显的相关性"
    elif strongest_absolute_value >= 0.3:
        overall_description = "最新窗口中部分因子与通胀观察到一定相关性"
    else:
        overall_description = "最新窗口中各因子与通胀的线性相关性整体偏弱"

    stronger_text = "、".join(
        format_correlation_item(ticker, float(latest_correlations[ticker]))
        for ticker in stronger_tickers
    )
    weaker_text = "、".join(
        format_correlation_item(ticker, float(latest_correlations[ticker]))
        for ticker in weaker_tickers
    )
    historical_text = "、".join(
        f"{ticker}（{FACTOR_NAMES[ticker]}，绝对相关性中位数="
        f"{historical_median_absolute[ticker]:.2f}）"
        for ticker in historically_higher_tickers
    )

    latest_cpi_date = monthly_data["CPI_YoY"].dropna().index[-1]
    latest_cpi_yoy = float(monthly_data.loc[latest_cpi_date, "CPI_YoY"])
    latest_indpro_date = monthly_data["INDPRO"].dropna().index[-1]
    latest_indpro = float(monthly_data.loc[latest_indpro_date, "INDPRO"])

    cpi_level_through_latest = monthly_data.loc[:latest_cpi_date, "CPIAUCSL"]
    missing_cpi_months = cpi_level_through_latest[
        cpi_level_through_latest.isna()
    ].index
    if len(missing_cpi_months) > 0 and latest_date < latest_cpi_date:
        missing_month_text = "、".join(
            date.strftime("%Y-%m") for date in missing_cpi_months
        )
        data_gap_note = (
            f"当前下载结果中 CPIAUCSL 在{missing_month_text}存在缺失；"
            "脚本未进行插值或前向填充，因此后续尚未重新积累12个完整观测，"
            f"最新完整相关窗口停留在{latest_date:%Y-%m}。"
        )
    else:
        data_gap_note = ""

    return (
        f"截至{latest_date:%Y-%m}可形成的最新12个月窗口，"
        f"{overall_description}。按相关系数绝对值比较，相对较高的是"
        f"{stronger_text}；相对较低的是{weaker_text}。"
        f"从全部滚动窗口的绝对相关性中位数看，{historical_text}相对靠前，"
        "但相关方向和强度会随阶段变化。"
        f"同期最新可得 CPI 同比为{latest_cpi_yoy:.2f}%（{latest_cpi_date:%Y-%m}），"
        f"工业生产指数为{latest_indpro:.2f}（{latest_indpro_date:%Y-%m}）。"
        f"{data_gap_note}"
        "上述结果仅是月度同步关系的描述性观察；12个月窗口较短，样本有限，"
        "且不代表因果关系，不构成可靠预测信号或投资建议。"
    )


def save_summary_text(summary: str, output_path: Path) -> None:
    """以 UTF-8 BOM 保存中文文本，兼容 Windows 记事本。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    temporary_path.write_text(summary + "\n", encoding="utf-8-sig")
    temporary_path.replace(output_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"状态标签 CSV（默认：{DEFAULT_LABELS_PATH}）",
    )
    parser.add_argument(
        "--zscore",
        type=Path,
        default=DEFAULT_ZSCORE_PATH,
        help=f"Z-score CSV（默认：{DEFAULT_ZSCORE_PATH}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIRECTORY,
        help=f"PNG 和 TXT 保存目录（默认：{SCRIPT_DIRECTORY}）",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    labels_path = arguments.labels.expanduser().resolve()
    zscore_path = arguments.zscore.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    chart_path = output_directory / CHART_FILENAME
    summary_path = output_directory / SUMMARY_FILENAME

    try:
        api_key = get_fred_api_key()
        selected_font = configure_chinese_font()
        daily_labels, daily_zscores = load_and_validate_factor_data(
            labels_path,
            zscore_path,
        )

        fred_client = Fred(api_key=api_key)
        industrial_production = download_fred_series(
            fred_client,
            "INDPRO",
            api_key,
        )
        cpi_index = download_fred_series(
            fred_client,
            "CPIAUCSL",
            api_key,
        )

        monthly_data = build_monthly_merged_data(
            daily_labels,
            daily_zscores,
            industrial_production,
            cpi_index,
        )
        rolling_correlations = calculate_rolling_correlations(monthly_data)
        create_comparison_chart(monthly_data, chart_path)
        summary = generate_correlation_summary(rolling_correlations, monthly_data)
        save_summary_text(summary, summary_path)
    except (
        FileNotFoundError,
        OSError,
        MacroAnalysisError,
        pd.errors.ParserError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print("FRED_API_KEY：已从环境变量读取（密钥内容未显示）")
    print(f"中文字体：{selected_font}")
    print(
        f"月度合并数据：{monthly_data.shape[0]} 行 × "
        f"{monthly_data.shape[1]} 列，"
        f"范围 {monthly_data.index.min():%Y-%m} 至 {monthly_data.index.max():%Y-%m}"
    )
    print(
        f"INDPRO：{industrial_production.index.min():%Y-%m} 至 "
        f"{industrial_production.index.max():%Y-%m}；"
        f"CPIAUCSL：{cpi_index.index.min():%Y-%m} 至 {cpi_index.index.max():%Y-%m}"
    )
    print(f"图表已保存：{chart_path}")
    print(f"总结已保存：{summary_path}")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
