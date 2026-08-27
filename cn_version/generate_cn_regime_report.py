"""生成 A 股 ETF 状态热力图和最新一期中文总结。

默认读取脚本同目录下的：
- cn_regime_labels.csv
- cn_regime_zscore.csv

默认输出：
- cn_regime_heatmap.png
- cn_latest_regime_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    # 使用非交互式后端，确保双击批处理或在终端运行时都能正常保存图片。
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import font_manager
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
except ImportError as exc:
    missing_package = getattr(exc, "name", "pandas/matplotlib")
    raise SystemExit(
        f"缺少依赖 {missing_package!r}。请先运行："
        "python -m pip install pandas matplotlib"
    ) from exc


ETF_TICKERS = [
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

# 热力图纵轴和文字总结统一使用这些中文名称。
ETF_NAMES = {
    "510300": "沪深300 大盘",
    "510880": "红利ETF 价值",
    "159915": "创业板 成长",
    "512100": "中证1000 小盘",
    "159928": "消费ETF 防御",
    "588000": "科创50 高弹性成长",
    "512010": "医药ETF",
    "512480": "半导体ETF",
    "512660": "军工ETF",
    "512880": "证券ETF",
    "515030": "新能源车ETF",
    "512800": "银行ETF 极致价值",
}

STATE_TO_VALUE = {
    "Contraction": -1,
    "Neutral": 0,
    "Expansion": 1,
}
STATE_TO_CHINESE = {
    "Contraction": "收缩",
    "Neutral": "中性",
    "Expansion": "扩张",
}

# A 股图表中使用红色表示扩张、蓝绿色表示收缩，中性使用浅灰色。
STATE_COLORS = {
    "Expansion": "#C95A5A",
    "Neutral": "#E1E5EA",
    "Contraction": "#4F7D8A",
}

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "Microsoft JhengHei",
    "PingFang SC",
    "Arial Unicode MS",
]

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_LABELS_PATH = SCRIPT_DIRECTORY / "cn_regime_labels.csv"
DEFAULT_ZSCORE_PATH = SCRIPT_DIRECTORY / "cn_regime_zscore.csv"
HEATMAP_FILENAME = "cn_regime_heatmap.png"
SUMMARY_FILENAME = "cn_latest_regime_summary.txt"
DEFAULT_RECENT_TRADING_DAYS = 252
DEFAULT_NEAR_THRESHOLD = 0.8


class ReportGenerationError(ValueError):
    """输入校验、图表生成或总结生成失败。"""


def configure_chinese_font() -> str:
    """从本机已安装字体中选择中文字体，避免 Matplotlib 中文乱码。"""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    raise ReportGenerationError(
        "未检测到可用中文字体。请安装微软雅黑、黑体或 Noto Sans CJK SC 后重试。"
    )


def read_csv_with_date_index(file_path: Path) -> pd.DataFrame:
    """读取以 Date 为索引的 CSV，并规范日期顺序。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{file_path}")

    dataframe = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
    dataframe.index = pd.to_datetime(dataframe.index, errors="coerce")

    if dataframe.index.isna().any():
        raise ReportGenerationError(f"{file_path.name} 含有无法识别的 Date。")
    if dataframe.index.duplicated().any():
        duplicated_date = dataframe.index[dataframe.index.duplicated()][0]
        raise ReportGenerationError(
            f"{file_path.name} 存在重复日期：{duplicated_date:%Y-%m-%d}"
        )

    dataframe = dataframe.sort_index()
    dataframe.index.name = "Date"
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    dataframe.columns.name = None

    if dataframe.empty:
        raise ReportGenerationError(f"输入文件没有可用数据：{file_path}")
    return dataframe


def load_and_validate_inputs(
    labels_path: Path,
    zscore_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取标签与 Z-score，并检查日期、列、状态值和最新一期完整性。"""
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
            raise ReportGenerationError(
                f"{file_name} 缺少 ETF 列：" + ", ".join(missing_tickers)
            )

    # 只按规定的 12 只 ETF 排序，避免额外列影响输出顺序。
    labels = labels.reindex(columns=ETF_TICKERS).apply(
        lambda column: column.astype("string").str.strip()
    )
    zscores = zscores.reindex(columns=ETF_TICKERS).apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not labels.index.equals(zscores.index):
        raise ReportGenerationError("两个输入文件的 Date 索引不完全一致。")

    allowed_states = set(STATE_TO_VALUE)
    invalid_state_mask = labels.notna() & ~labels.isin(allowed_states)
    if invalid_state_mask.any().any():
        row_position, column_position = np.argwhere(
            invalid_state_mask.to_numpy()
        )[0]
        invalid_date = labels.index[row_position]
        invalid_ticker = labels.columns[column_position]
        invalid_value = labels.iloc[row_position, column_position]
        raise ReportGenerationError(
            f"发现无效状态：{invalid_date:%Y-%m-%d} "
            f"{invalid_ticker}={invalid_value!r}"
        )

    # 最新一期用于自动总结，要求 12 只 ETF 的标签和 Z-score 均完整。
    if labels.iloc[-1].isna().any() or zscores.iloc[-1].isna().any():
        raise ReportGenerationError("最新一期存在空标签或空 Z-score，无法生成总结。")

    return labels, zscores.astype(float)


def select_recent_complete_rows(
    labels: pd.DataFrame,
    recent_trading_days: int,
) -> pd.DataFrame:
    """选择最近指定数量、12只ETF标签均完整的交易日。"""
    if recent_trading_days <= 0:
        raise ReportGenerationError("--recent-trading-days 必须是正整数。")

    complete_labels = labels.loc[labels.notna().all(axis=1)]
    if len(complete_labels) < recent_trading_days:
        raise ReportGenerationError(
            f"12只ETF均有状态的数据只有 {len(complete_labels)} 行，"
            f"不足 {recent_trading_days} 个交易日。"
        )
    return complete_labels.tail(recent_trading_days)


def create_regime_heatmap(
    recent_labels: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制横轴为时间、纵轴为 12 只 ETF 的离散状态热力图。"""
    encoded_states = recent_labels.apply(
        lambda column: column.map(STATE_TO_VALUE)
    ).transpose()

    color_map = ListedColormap(
        [
            STATE_COLORS["Contraction"],
            STATE_COLORS["Neutral"],
            STATE_COLORS["Expansion"],
        ]
    )
    color_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)

    figure, axis = plt.subplots(figsize=(17, 8.2), dpi=180)
    axis.imshow(
        encoded_states.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=color_norm,
    )

    # 横轴只显示少量等距日期刻度，避免 252 个日期相互重叠。
    tick_count = min(9, len(recent_labels))
    tick_positions = np.unique(
        np.linspace(0, len(recent_labels) - 1, num=tick_count, dtype=int)
    )
    tick_labels = [
        recent_labels.index[position].strftime("%Y-%m-%d")
        for position in tick_positions
    ]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=28, ha="right")

    axis.set_yticks(np.arange(len(ETF_TICKERS)))
    axis.set_yticklabels([ETF_NAMES[ticker] for ticker in ETF_TICKERS])
    axis.set_xlabel("交易日期")
    axis.set_ylabel("A股风格与行业代理 ETF")

    start_date = recent_labels.index[0].strftime("%Y-%m-%d")
    end_date = recent_labels.index[-1].strftime("%Y-%m-%d")
    figure.suptitle(
        f"A股 ETF 状态热力图（最近 {len(recent_labels)} 个交易日）\n"
        f"{start_date} 至 {end_date}",
        y=0.965,
        fontsize=17,
        fontweight="normal",
    )

    # 用白色横线分隔 ETF，帮助快速追踪单个品种的状态变化。
    for boundary in np.arange(-0.5, len(ETF_TICKERS), 1.0):
        axis.axhline(boundary, color="white", linewidth=1.25)

    legend_handles = [
        Patch(
            facecolor=STATE_COLORS[state],
            edgecolor="none",
            label=f"{state}（{STATE_TO_CHINESE[state]}）",
        )
        for state in ("Expansion", "Neutral", "Contraction")
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    axis.tick_params(axis="both", length=0, labelsize=10.5)
    for spine in axis.spines.values():
        spine.set_color("#8B95A5")
        spine.set_linewidth(0.8)

    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.subplots_adjust(left=0.20, right=0.985, bottom=0.15, top=0.79)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    figure.savefig(
        temporary_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        format="png",
    )
    plt.close(figure)
    temporary_path.replace(output_path)


def determine_overall_assessment(latest_labels: pd.Series) -> str:
    """根据当前组合生成保守、可重复的 A 股风格判断。"""
    state_score = {"Expansion": 1, "Neutral": 0, "Contraction": -1}
    expansion_count = int(latest_labels.eq("Expansion").sum())
    contraction_count = int(latest_labels.eq("Contraction").sum())

    value_defense_tickers = ["510880", "159928", "512010", "512800"]
    growth_tickers = ["159915", "588000", "512480", "515030"]
    broad_market_tickers = ["510300", "512100"]

    value_defense_score = sum(
        state_score[str(latest_labels[ticker])]
        for ticker in value_defense_tickers
    )
    growth_score = sum(
        state_score[str(latest_labels[ticker])] for ticker in growth_tickers
    )
    broad_market_score = sum(
        state_score[str(latest_labels[ticker])]
        for ticker in broad_market_tickers
    )

    if value_defense_score >= 2 and (
        growth_score <= -2 or broad_market_score <= -1
    ):
        return "防御与价值方向相对占优，而成长和宽基方向偏弱，行业轮动与风格分化较为明显"
    if expansion_count >= 2 and contraction_count >= 2:
        return "强弱信号并存，行业轮动与风格分化较为明显"
    if expansion_count >= 6:
        return "扩张状态占优，多个风格和行业方向同步走强"
    if contraction_count >= 6:
        return "收缩状态占优，多数风格和行业方向偏弱"
    if expansion_count > contraction_count:
        return "局部方向走强，但尚未形成广泛一致的扩张格局"
    if contraction_count > expansion_count:
        return "局部方向走弱，但尚未形成广泛一致的收缩格局"
    return "整体以中性为主，暂未形成明确的单边风格"


def describe_etf_state(
    ticker: str,
    current_label: str,
    previous_label: object,
    zscore: float,
    near_threshold: float,
) -> str:
    """生成单只 ETF 的中文状态描述，并提示最新一期的状态切换。"""
    etf_label = f"{ETF_NAMES[ticker]}（{ticker}）"
    current_chinese = STATE_TO_CHINESE[current_label]

    if pd.notna(previous_label) and str(previous_label) != current_label:
        previous_chinese = STATE_TO_CHINESE[str(previous_label)]
        state_phrase = f"由{previous_chinese}转为{current_chinese}状态"
    else:
        state_phrase = f"处于{current_chinese}状态"

    if current_label == "Expansion":
        interpretation = (
            "明显高于过去252个交易日的历史水平"
            if zscore >= 2.0
            else "高于过去252个交易日的历史水平"
        )
    elif current_label == "Contraction":
        interpretation = (
            "明显低于过去252个交易日的历史水平"
            if zscore <= -2.0
            else "低于过去252个交易日的历史水平"
        )
    elif zscore >= near_threshold:
        interpretation = "仍属中性，但已接近扩张阈值，值得关注"
    else:
        interpretation = "仍属中性，但已接近收缩阈值，值得关注"

    return (
        f"{etf_label}{state_phrase}（Z-score={zscore:.2f}），"
        f"{interpretation}"
    )


def generate_latest_summary(
    labels: pd.DataFrame,
    zscores: pd.DataFrame,
    near_threshold: float,
) -> str:
    """读取最新一期状态，自动生成 A 股风格与行业轮动总结。"""
    if not 0 < near_threshold <= 1:
        raise ReportGenerationError("--near-threshold 必须大于 0 且不超过 1。")

    latest_date = labels.index[-1]
    latest_labels = labels.iloc[-1]
    latest_zscores = zscores.iloc[-1]
    previous_labels = labels.iloc[-2] if len(labels) >= 2 else latest_labels

    expansion_count = int(latest_labels.eq("Expansion").sum())
    neutral_count = int(latest_labels.eq("Neutral").sum())
    contraction_count = int(latest_labels.eq("Contraction").sum())
    overall_assessment = determine_overall_assessment(latest_labels)

    non_neutral_tickers = [
        ticker for ticker in ETF_TICKERS if latest_labels[ticker] != "Neutral"
    ]
    near_threshold_neutral_tickers = [
        ticker
        for ticker in ETF_TICKERS
        if latest_labels[ticker] == "Neutral"
        and abs(float(latest_zscores[ticker])) >= near_threshold
    ]

    tickers_to_explain = non_neutral_tickers + near_threshold_neutral_tickers
    details = [
        describe_etf_state(
            ticker=ticker,
            current_label=str(latest_labels[ticker]),
            previous_label=previous_labels[ticker],
            zscore=float(latest_zscores[ticker]),
            near_threshold=near_threshold,
        )
        for ticker in tickers_to_explain
    ]

    if details:
        detail_text = "；\n".join(details) + "。"
    else:
        detail_text = "12只ETF均处于中性状态，且尚未接近状态切换阈值。"

    return (
        f"截至{latest_date:%Y-%m-%d}，当前A股ETF状态呈现{overall_assessment}。"
        f"12只ETF中，{expansion_count}只处于扩张、{neutral_count}只处于中性、"
        f"{contraction_count}只处于收缩。\n"
        f"具体来看：{detail_text}\n"
        "以上内容仅是对滚动收益率状态的客观描述，不构成投资建议。"
    )


def save_summary_text(summary: str, output_path: Path) -> None:
    """用 UTF-8 BOM 保存中文文本，兼容 Windows 记事本。"""
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
    parser.add_argument(
        "--recent-trading-days",
        type=int,
        default=DEFAULT_RECENT_TRADING_DAYS,
        help=f"热力图展示的最近交易日数量（默认：{DEFAULT_RECENT_TRADING_DAYS}）",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=DEFAULT_NEAR_THRESHOLD,
        help=f"中性状态接近阈值的提示线（默认：±{DEFAULT_NEAR_THRESHOLD:g}）",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    labels_path = arguments.labels.expanduser().resolve()
    zscore_path = arguments.zscore.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    heatmap_path = output_directory / HEATMAP_FILENAME
    summary_path = output_directory / SUMMARY_FILENAME

    try:
        selected_font = configure_chinese_font()
        labels, zscores = load_and_validate_inputs(labels_path, zscore_path)
        recent_labels = select_recent_complete_rows(
            labels,
            arguments.recent_trading_days,
        )
        create_regime_heatmap(recent_labels, heatmap_path)
        summary = generate_latest_summary(
            labels,
            zscores,
            arguments.near_threshold,
        )
        save_summary_text(summary, summary_path)
    except (
        FileNotFoundError,
        OSError,
        ReportGenerationError,
        pd.errors.ParserError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"中文字体：{selected_font}")
    print(
        f"热力图范围：{recent_labels.index[0]:%Y-%m-%d} 至 "
        f"{recent_labels.index[-1]:%Y-%m-%d}，"
        f"共 {len(recent_labels)} 个交易日。"
    )
    print(f"热力图已保存：{heatmap_path}")
    print(f"总结已保存：{summary_path}")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
