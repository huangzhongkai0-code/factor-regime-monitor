"""生成最近一年因子状态热力图和最新一期中文状态总结。

默认读取脚本同目录下的：
- regime_labels.csv
- regime_zscore.csv

默认输出：
- regime_heatmap.png
- latest_regime_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    # 使用非交互式后端，确保脚本可在终端、服务器和自动化任务中运行。
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


ETF_TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
FACTOR_NAMES = {
    "SPY": "大盘",
    "IVE": "价值因子",
    "IVW": "成长因子",
    "MTUM": "动量因子",
    "USMV": "低波动因子",
    "IWM": "小盘股因子",
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
STATE_COLORS = {
    "Expansion": "#2E8B67",
    "Neutral": "#D9DEE7",
    "Contraction": "#C85A63",
}

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
HEATMAP_FILENAME = "regime_heatmap.png"
SUMMARY_FILENAME = "latest_regime_summary.txt"
DEFAULT_RECENT_TRADING_DAYS = 252
DEFAULT_NEAR_THRESHOLD = 0.8


class ReportGenerationError(ValueError):
    """输入校验、图表生成或总结生成失败。"""


def configure_chinese_font() -> str:
    """选择本机已安装的中文字体，避免 Matplotlib 中文乱码。"""
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
    dataframe = dataframe.loc[~dataframe.index.isna()].sort_index()
    dataframe.index.name = "Date"
    dataframe.columns.name = None

    if dataframe.empty:
        raise ReportGenerationError(f"输入文件没有可用数据：{file_path}")
    if dataframe.index.duplicated().any():
        duplicated_date = dataframe.index[dataframe.index.duplicated()][0]
        raise ReportGenerationError(
            f"{file_path.name} 存在重复日期：{duplicated_date:%Y-%m-%d}"
        )
    return dataframe


def load_and_validate_inputs(
    labels_path: Path,
    zscore_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取标签与 Z-score，并严格检查日期、列和状态值是否一致。"""
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
        raise ReportGenerationError("两个输入文件的 Date 索引不完全一致。")

    allowed_states = set(STATE_TO_VALUE)
    invalid_state_mask = labels.notna() & ~labels.isin(allowed_states)
    if invalid_state_mask.any().any():
        row_position, column_position = np.argwhere(invalid_state_mask.to_numpy())[0]
        invalid_date = labels.index[row_position]
        invalid_ticker = labels.columns[column_position]
        invalid_value = labels.iloc[row_position, column_position]
        raise ReportGenerationError(
            f"发现无效状态：{invalid_date:%Y-%m-%d} {invalid_ticker}={invalid_value!r}"
        )

    # 最新一期是文字总结的基础，要求六个因子的标签和 Z-score 均完整。
    if labels.iloc[-1].isna().any() or zscores.iloc[-1].isna().any():
        raise ReportGenerationError("最新一期存在空标签或空 Z-score，无法生成总结。")

    return labels, zscores.astype(float)


def select_recent_complete_rows(
    labels: pd.DataFrame,
    recent_trading_days: int,
) -> pd.DataFrame:
    """选择最近指定数量、六个因子标签均完整的交易日。"""
    if recent_trading_days <= 0:
        raise ReportGenerationError("--recent-trading-days 必须是正整数。")

    complete_labels = labels.loc[labels.notna().all(axis=1)]
    if len(complete_labels) < recent_trading_days:
        raise ReportGenerationError(
            f"完整状态数据只有 {len(complete_labels)} 行，"
            f"不足 {recent_trading_days} 个交易日。"
        )
    return complete_labels.tail(recent_trading_days)


def create_regime_heatmap(
    recent_labels: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制“横轴时间、纵轴因子”的离散状态热力图。"""
    encoded_states = recent_labels.apply(
        lambda column: column.map(STATE_TO_VALUE)
    ).transpose()

    # 按 -1/0/1 顺序配置红、灰、绿三种离散颜色。
    color_map = ListedColormap(
        [
            STATE_COLORS["Contraction"],
            STATE_COLORS["Neutral"],
            STATE_COLORS["Expansion"],
        ]
    )
    color_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)

    figure, axis = plt.subplots(figsize=(15, 5.6), dpi=180)
    axis.imshow(
        encoded_states.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=color_norm,
    )

    # 使用少量等距日期刻度，避免 252 个日期挤在一起。
    tick_count = min(8, len(recent_labels))
    tick_positions = np.unique(
        np.linspace(0, len(recent_labels) - 1, num=tick_count, dtype=int)
    )
    tick_labels = [
        recent_labels.index[position].strftime("%Y-%m-%d")
        for position in tick_positions
    ]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=28, ha="right")

    factor_axis_labels = [
        f"{ticker}  {FACTOR_NAMES[ticker]}" for ticker in ETF_TICKERS
    ]
    axis.set_yticks(np.arange(len(ETF_TICKERS)))
    axis.set_yticklabels(factor_axis_labels)
    axis.set_xlabel("交易日期")
    axis.set_ylabel("因子 ETF")

    start_date = recent_labels.index[0].strftime("%Y-%m-%d")
    end_date = recent_labels.index[-1].strftime("%Y-%m-%d")
    figure.suptitle(
        f"因子 ETF 状态热力图（最近 {len(recent_labels)} 个交易日）\n"
        f"{start_date} 至 {end_date}",
        y=0.97,
        fontweight="normal",
    )

    # 横向白色分隔线帮助区分六个因子，不添加密集纵向网格。
    for boundary in np.arange(-0.5, len(ETF_TICKERS), 1.0):
        axis.axhline(boundary, color="white", linewidth=1.2)

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
        bbox_to_anchor=(0.5, 1.26),
        ncol=3,
        frameon=False,
    )

    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_color("#8B95A5")
        spine.set_linewidth(0.8)

    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.74)

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


def determine_overall_assessment(
    latest_labels: pd.Series,
    latest_zscores: pd.Series,
    near_threshold: float,
) -> str:
    """根据最新因子组合生成简洁、可重复的整体判断。"""
    expansion_factors = latest_labels.index[latest_labels.eq("Expansion")].tolist()
    contraction_factors = latest_labels.index[
        latest_labels.eq("Contraction")
    ].tolist()

    # 低波动走强、同时风险敏感因子偏弱时，优先判定为防御性风格。
    risk_sensitive_factors = ["IVW", "MTUM", "IWM"]
    risk_sensitive_weak = any(
        latest_labels[ticker] == "Contraction"
        or latest_zscores[ticker] <= -near_threshold
        for ticker in risk_sensitive_factors
    )
    if latest_labels["USMV"] == "Expansion" and risk_sensitive_weak:
        return "防御性偏强，且风格分化较为明显"
    if expansion_factors and contraction_factors:
        return "多空风格并存，轮动与分化明显"
    if len(expansion_factors) >= 3:
        return "多数因子偏强，风险偏好有所扩张"
    if len(contraction_factors) >= 3:
        return "多数因子偏弱，市场风格整体收缩"
    if expansion_factors:
        return "局部因子走强，但整体仍以中性为主"
    if contraction_factors:
        return "局部因子走弱，但整体仍以中性为主"
    return "整体处于中性区间，尚未形成明确单边风格"


def describe_factor_state(
    ticker: str,
    current_label: str,
    previous_label: object,
    zscore: float,
    near_threshold: float,
) -> str:
    """生成单个因子的中文状态说明，并识别是否发生状态切换。"""
    factor_label = f"{ticker}（{FACTOR_NAMES[ticker]}）"
    current_chinese = STATE_TO_CHINESE[current_label]

    if pd.notna(previous_label) and previous_label != current_label:
        previous_chinese = STATE_TO_CHINESE[str(previous_label)]
        state_phrase = f"由{previous_chinese}转为{current_chinese}状态"
    else:
        state_phrase = f"处于{current_chinese}状态"

    if current_label == "Expansion":
        interpretation = "显著强于过去252个交易日的历史水平"
    elif current_label == "Contraction":
        interpretation = "显著弱于过去252个交易日的历史水平"
    elif zscore >= near_threshold:
        interpretation = "仍属中性，但已接近扩张阈值，值得关注"
    else:
        interpretation = "仍属中性，但已接近收缩阈值，值得关注"

    return (
        f"{factor_label}{state_phrase}（Z-score={zscore:.2f}），"
        f"{interpretation}"
    )


def generate_latest_summary(
    labels: pd.DataFrame,
    zscores: pd.DataFrame,
    near_threshold: float,
) -> str:
    """读取最新一期数据，自动生成中文状态总结。"""
    if near_threshold <= 0:
        raise ReportGenerationError("--near-threshold 必须大于 0。")

    latest_date = labels.index[-1]
    latest_labels = labels.iloc[-1]
    previous_labels = labels.iloc[-2] if len(labels) >= 2 else latest_labels
    latest_zscores = zscores.iloc[-1]

    overall_assessment = determine_overall_assessment(
        latest_labels,
        latest_zscores,
        near_threshold,
    )

    non_neutral_factors = [
        ticker for ticker in ETF_TICKERS if latest_labels[ticker] != "Neutral"
    ]
    near_threshold_neutral_factors = [
        ticker
        for ticker in ETF_TICKERS
        if latest_labels[ticker] == "Neutral"
        and abs(float(latest_zscores[ticker])) >= near_threshold
    ]

    # 先说明明确处于扩张/收缩的因子，再提示接近阈值的中性因子。
    factors_to_explain = non_neutral_factors + near_threshold_neutral_factors
    factor_details: list[str] = []
    for ticker in factors_to_explain:
        current_label = str(latest_labels[ticker])
        current_zscore = float(latest_zscores[ticker])

        factor_details.append(
            describe_factor_state(
                ticker=ticker,
                current_label=current_label,
                previous_label=previous_labels[ticker],
                zscore=current_zscore,
                near_threshold=near_threshold,
            )
        )

    if factor_details:
        detail_text = "；\n".join(factor_details) + "。"
    else:
        detail_text = "六个因子均处于中性状态，且尚未接近状态切换阈值。"

    return (
        f"截至{latest_date:%Y-%m-%d}，市场风格呈现{overall_assessment}。\n"
        f"具体来看：{detail_text}"
    )


def save_summary_text(summary: str, output_path: Path) -> None:
    """用 UTF-8 BOM 保存中文文本，兼容 Windows 记事本和 Excel。"""
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
        help=f"中性因子接近阈值的提示线（默认：±{DEFAULT_NEAR_THRESHOLD:g}）",
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
