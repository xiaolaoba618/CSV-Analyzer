import atexit
import shutil
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
MAX_CATEGORIES = 15
MAX_HEATMAP_COLUMNS = 30
MAX_SCATTER_POINTS = 10000
MAX_SCATTER_VARIABLES = 10

def configure_chinese_font():
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS"
    ]
    available_fonts = {font.name for font in plt.matplotlib.font_manager.fontManager.ttflist}
    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break
    if selected_font is not None:
        plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return selected_font

# 图表生成后只会被报告模块（report/generator.py、report/dashboard_generator.py）
# 读取一次，然后转成 base64 内嵌到 HTML 中。因此图表本身不需要长期驻留在本地。
# 这里统一把图表写入系统临时目录，并在进程退出时自动删除，
# 这样既不影响报告生成，也不会在项目目录里残留 PNG 文件。
_FIGURE_DIRECTORY = None

def _remove_figure_directory(directory):
    """删除图表临时目录；失败时静默忽略，不影响主流程。"""
    shutil.rmtree(directory, ignore_errors=True)

def _get_figure_directory():
    """返回本进程独占的图表临时目录，首次调用时创建并注册退出清理。"""
    global _FIGURE_DIRECTORY
    if _FIGURE_DIRECTORY is None:
        _FIGURE_DIRECTORY = tempfile.mkdtemp(prefix="csv_analyzer_figures_")
        atexit.register(_remove_figure_directory, _FIGURE_DIRECTORY)
    return Path(_FIGURE_DIRECTORY)

def _ensure_output_dir(output_dir=None):
    """返回图表输出目录。

    保留原有函数名与参数以便兼容既有调用，但不再使用传入的 output_dir：
    图表统一写入临时目录，避免在本地留下 PNG 文件。
    """
    return _get_figure_directory()

def _safe_filename(name):
    invalid_characters = '<>:"/\\|?*'
    result = str(name)
    for character in invalid_characters:
        result = result.replace(character, "_")
    result = result.strip()
    if not result:
        result = "unnamed"
    return result

def _sample_series(series, max_points=MAX_SCATTER_POINTS):
    series = series.dropna()
    if len(series) <= max_points:
        return series
    return series.sample(n=max_points, random_state=42)

def _sample_dataframe(df, max_points=MAX_SCATTER_POINTS):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42)

def _get_columns_by_type(df, variable_types, allowed_types):
    if variable_types is None:
        return []
    return [column for column, info in variable_types.items() if (column in df.columns and info["detected_type"] in allowed_types)]

def _get_numeric_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    return _get_columns_by_type(df, variable_types, {"numeric"})

def _get_categorical_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return _get_columns_by_type(df, variable_types, {"categorical", "categorical_numeric", "binary", "boolean"})

def _add_axis_grid(ax):
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

def _add_value_labels(ax, values, decimals=0):
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        if decimals == 0:
            label = f"{value:,.0f}"
        else:
            label = f"{value:,.{decimals}f}"
        ax.text(index, value, label, ha="center", va="bottom", fontsize=8, alpha=0.85)

def _add_horizontal_value_labels(ax, values):
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        ax.text(
            value,
            index,
            f"{value:,.0f}",
            va="center",
            ha="left",
            fontsize=8,
            alpha=0.85
        )

def create_histogram(df, column, output_dir, bins=30):
    output_dir = _ensure_output_dir(output_dir)
    series = (df[column].replace([np.inf, -np.inf], np.nan).dropna())
    if len(series) == 0:
        return None
    series = _sample_series(series)
    if series.nunique() <= 1:
        return None
    mean_value = series.mean()
    median_value = series.median()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(series, bins=bins, alpha=0.78, edgecolor="white", linewidth=0.7)
    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.2f}"
    )
    ax.axvline(
        median_value,
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {median_value:.2f}"
    )
    ax.set_xlabel(column)
    ax.set_ylabel("频数")
    ax.set_title(
        f"{column} 分布"
    )
    _add_axis_grid(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = (
        output_dir
        / f"{_safe_filename(column)}_hist.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def create_boxplot(df, column, output_dir):
    output_dir = _ensure_output_dir(output_dir)
    series = (df[column].replace([np.inf, -np.inf], np.nan).dropna())
    if len(series) == 0:
        return None
    if series.nunique() <= 1:
        return None
    series = _sample_series(series)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.boxplot(
        series,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
        medianprops={"linewidth": 2},
        meanprops={"marker": "D", "markersize": 5}
    )
    ax.set_ylabel(column)
    ax.set_title(
        f"{column} 箱线图"
    )
    ax.set_xticks([1])
    ax.set_xticklabels([column])
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    output_path = (
        output_dir
        / f"{_safe_filename(column)}_box.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def create_categorical_bar(df, column, output_dir, max_categories=MAX_CATEGORIES):
    output_dir = _ensure_output_dir(output_dir)
    series = df[column].dropna()
    if len(series) == 0:
        return None
    value_counts = (series.astype(str).value_counts().head(max_categories))
    if len(value_counts) == 0:
        return None
    value_counts = (value_counts.sort_values())
    fig_height = max(5, len(value_counts) * 0.38)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bars = ax.barh(range(len(value_counts)), value_counts.values, alpha=0.82)
    ax.set_yticks(range(len(value_counts)))
    ax.set_yticklabels(value_counts.index)
    ax.set_xlabel("数量")
    ax.set_ylabel(column)
    ax.set_title(
        f"{column} 类别分布"
    )
    _add_axis_grid(ax)
    for bar, value in zip(bars, value_counts.values):
        ax.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=8
        )
    fig.tight_layout()
    output_path = (
        output_dir
        / f"{_safe_filename(column)}_bar.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def create_scatter(df, column_x, column_y, output_dir, max_points=MAX_SCATTER_POINTS):
    output_dir = _ensure_output_dir(output_dir)
    pair = df[[column_x, column_y]].copy()
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return None
    if (pair[column_x].nunique() <= 1 or pair[column_y].nunique() <= 1):
        return None
    pair = _sample_dataframe(pair, max_points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pair[column_x], pair[column_y], alpha=0.42, s=22, edgecolors="none")
    try:
        x = pair[column_x].to_numpy()
        y = pair[column_y].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = (slope * x_line + intercept)
        ax.plot(x_line, y_line, linestyle="--", linewidth=1.8, label="线性趋势")
        ax.legend(frameon=False)
    except Exception:
        pass
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)
    ax.set_title(
        f"{column_x} 与 {column_y} 的关系"
    )
    ax.grid(alpha=0.18, linestyle="--")
    fig.tight_layout()
    filename = (
        f"{_safe_filename(column_x)}"
        f"_vs_"
        f"{_safe_filename(column_y)}"
        f"_scatter.png"
    )
    output_path = (output_dir / filename)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def create_correlation_heatmap(correlation_result, output_dir, method="spearman", max_columns=MAX_HEATMAP_COLUMNS):
    output_dir = _ensure_output_dir(output_dir)
    if not correlation_result:
        return None
    if method not in correlation_result:
        return None
    method_result = correlation_result[method]
    if not method_result:
        return None
    correlation_matrix = method_result.get("correlation")
    if correlation_matrix is None:
        return None
    if correlation_matrix.empty:
        return None
    if len(correlation_matrix.columns) == 0:
        return None
    columns = correlation_matrix.columns
    if len(columns) > max_columns:
        columns = columns[:max_columns]
        correlation_matrix = (correlation_matrix.loc[columns, columns])
    n_columns = len(columns)
    figure_size = max(8, n_columns * 0.5)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(correlation_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(n_columns))
    ax.set_yticklabels(columns)
    for i in range(n_columns):
        for j in range(n_columns):
            value = correlation_matrix.iloc[i, j]
            if pd.isna(value):
                continue
            text_color = ("white" if abs(value) >= 0.55 else "black")
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color
            )
    ax.set_title(
        f"{method.title()} 相关系数热力图"
    )
    fig.tight_layout()
    output_path = (
        output_dir
        / f"{method}_correlation_heatmap.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def plot_hourly_distribution(time_analysis, output_dir=None):
    output_dir = _ensure_output_dir()
    paths = []
    analyses = time_analysis.get("analyses", {})
    for column, analysis in analyses.items():
        hourly = analysis.get("hourly_distribution", {})
        if hourly.get("status") != "ok":
            continue
        results = hourly.get("results", [])
        if not results:
            continue
        hours = [item["hour"] for item in results]
        counts = [item["count"] for item in results]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(hours, counts, alpha=0.82)
        ax.set_xticks(range(24))
        ax.set_xlabel("小时")
        ax.set_ylabel("观测数量")
        ax.set_title(
            f"{column} 小时分布"
        )
        _add_axis_grid(ax)
        max_index = int(np.argmax(counts))
        bars[max_index].set_alpha(1.0)
        fig.tight_layout()
        output_path = (
            output_dir
            / f"{_safe_filename(column)}_hourly.png"
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(output_path))
    return paths

def plot_weekday_distribution(time_analysis, output_dir=None):
    output_dir = _ensure_output_dir()
    paths = []
    analyses = time_analysis.get("analyses", {})
    for column, analysis in analyses.items():
        weekday = analysis.get("weekday_distribution", {})
        if weekday.get("status") != "ok":
            continue
        results = weekday.get("results", [])
        if not results:
            continue
        labels = [item["weekday"] for item in results]
        counts = [item["count"] for item in results]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, counts, alpha=0.82)
        ax.set_xlabel("星期")
        ax.set_ylabel("观测数量")
        ax.set_title(
            f"{column} 星期分布"
        )
        ax.tick_params(axis="x", rotation=20)
        _add_axis_grid(ax)
        _add_value_labels(ax, counts)
        fig.tight_layout()
        output_path = (
            output_dir
            / f"{_safe_filename(column)}_weekday.png"
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(output_path))
    return paths

def plot_daily_distribution(time_analysis, output_dir=None):
    output_dir = _ensure_output_dir()
    paths = []
    analyses = time_analysis.get("analyses", {})
    for column, analysis in analyses.items():
        daily = analysis.get("daily_distribution", {})
        if daily.get("status") != "ok":
            continue
        results = daily.get("results", [])
        if not results:
            continue
        dates = pd.to_datetime([item["date"] for item in results], errors="coerce")
        counts = [item["count"] for item in results]
        valid_mask = ~dates.isna()
        dates = dates[valid_mask]
        counts = np.array(counts)[valid_mask]
        if len(dates) == 0:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(dates, counts, marker="o", markersize=3.5, linewidth=1.8)
        ax.fill_between(dates, counts, alpha=0.10)
        ax.set_xlabel("日期")
        ax.set_ylabel("观测数量")
        ax.set_title(
            f"{column} 每日观测趋势"
        )
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.18, linestyle="--")
        fig.autofmt_xdate()
        fig.tight_layout()
        output_path = (
            output_dir
            / f"{_safe_filename(column)}_daily.png"
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(output_path))
    return paths

def plot_time_series(time_series_analysis, output_dir, max_variables=10):
    if not time_series_analysis:
        return []
    if (time_series_analysis.get("status") != "ok"):
        return []
    aggregation = (time_series_analysis.get("aggregation"))
    if not aggregation:
        return []
    if aggregation.get("status") != "ok":
        return []
    variables = aggregation.get("variables", {})
    if not variables:
        return []
    output_dir = _ensure_output_dir(output_dir)
    configure_chinese_font()
    frequency = aggregation.get("frequency", "unknown")
    paths = []
    valid_variables = [column for column, result in variables.items() if result.get("status") == "ok"]
    for column in valid_variables[:max_variables]:
        try:
            result = variables[column]
            records = result.get("results", [])
            if not records:
                continue
            data = pd.DataFrame(records)
            fig, ax = plt.subplots(figsize=(11, 5.5))
            if "time_index" in data.columns:
                data["time_index"] = pd.to_numeric(data["time_index"], errors="coerce")
                data = data.dropna(subset=["time_index"])
                value_key = "value" if frequency == "irregular" else "mean"
                if value_key not in data.columns or data.empty:
                    plt.close(fig)
                    continue
                data[value_key] = pd.to_numeric(data[value_key], errors="coerce")
                data = data.dropna(subset=[value_key])
                if data.empty:
                    plt.close(fig)
                    continue
                ax.plot(data["time_index"], data[value_key], marker="o", markersize=3, linewidth=1.5)
                ax.set_title(f"{column} 时间序列")
                ax.set_xlabel("时间序号")
                ax.set_ylabel(str(column))
                ax.grid(alpha=0.18, linestyle="--")
                filename = _safe_filename(f"time_series_{column}.png")
                output_path = output_dir / filename
                fig.tight_layout()
                fig.savefig(output_path, bbox_inches="tight")
                plt.close(fig)
                paths.append(str(output_path))
                continue
            if "datetime" not in data.columns:
                plt.close(fig)
                continue
            data["datetime"] = (pd.to_datetime(data["datetime"], errors="coerce"))
            data = data.dropna(subset=["datetime"])
            if data.empty:
                continue
            if frequency == "irregular":
                if "value" not in data.columns:
                    plt.close(fig)
                    continue
                data["value"] = pd.to_numeric(data["value"], errors="coerce")
                data = data.dropna(subset=["value"])
                if data.empty:
                    plt.close(fig)
                    continue
                ax.plot(data["datetime"], data["value"], marker="o", markersize=3, linewidth=1.2, alpha=0.8)
            else:
                if "mean" not in data.columns:
                    plt.close(fig)
                    continue
                data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
                data = data.dropna(subset=["mean"])
                if data.empty:
                    plt.close(fig)
                    continue
                ax.plot(data["datetime"], data["mean"], linewidth=2)
                overall_mean = data["mean"].mean()
                ax.axhline(
                    overall_mean,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.65,
                    label=(
                        f"Mean: "
                        f"{overall_mean:.2f}"
                    )
                )
                ax.legend(frameon=False)
            ax.set_title(
                f"{column} 时间序列"
            )
            ax.set_xlabel("时间")
            ax.set_ylabel(str(column))
            ax.grid(alpha=0.18, linestyle="--")
            fig.autofmt_xdate()
            filename = _safe_filename(
                f"time_series_{column}.png"
            )
            output_path = (output_dir / filename)
            fig.tight_layout()
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(output_path))
        except Exception:
            plt.close("all")
            continue
    return paths

def create_all_visualizations(
    df,
    output_dir,
    correlation_result=None,
    correlation_method="spearman",
    variable_types=None,
    time_analysis=None,
    time_series_analysis=None
):
    configure_chinese_font()
    output_dir = _ensure_output_dir(output_dir)
    numeric_columns = (_get_numeric_columns(df, variable_types))
    categorical_columns = (_get_categorical_columns(df, variable_types))
    histograms = []
    boxplots = []
    categorical_bars = []
    scatter_plots = []
    for column in numeric_columns:
        histogram = create_histogram(df, column, output_dir)
        if histogram is not None:
            histograms.append(histogram)
        boxplot = create_boxplot(df, column, output_dir)
        if boxplot is not None:
            boxplots.append(boxplot)
    for column in categorical_columns:
        chart = create_categorical_bar(df, column, output_dir)
        if chart is not None:
            categorical_bars.append(chart)
    scatter_columns = numeric_columns[:MAX_SCATTER_VARIABLES]
    for i in range(len(scatter_columns)):
        for j in range(i + 1, len(scatter_columns)):
            scatter = create_scatter(df, scatter_columns[i], scatter_columns[j], output_dir)
            if scatter is not None:
                scatter_plots.append(scatter)
    heatmap = None
    if correlation_result is not None:
        heatmap = (create_correlation_heatmap(correlation_result, output_dir, method=correlation_method))
    hourly_paths = []
    weekday_paths = []
    daily_paths = []
    if time_analysis:
        hourly_paths = (plot_hourly_distribution(time_analysis, output_dir))
        weekday_paths = (plot_weekday_distribution(time_analysis, output_dir))
        daily_paths = (plot_daily_distribution(time_analysis, output_dir))
    time_series_paths = []
    if time_series_analysis:
        time_series_paths = (plot_time_series(time_series_analysis, output_dir))
    return {
        "histograms": histograms,
        "boxplots": boxplots,
        "categorical_bars": categorical_bars,
        "scatter_plots": scatter_plots,
        "correlation_heatmap": heatmap,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "time_hourly": hourly_paths,
        "time_weekday": weekday_paths,
        "time_daily": daily_paths,
        "time_series": time_series_paths
    }
