from pathlib import Path
import base64
import html
TEMPLATE_PATH = Path(__file__).parent / "template.html"
DEFAULT_REPORT_PATH = Path("output/report.html")

def format_number(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)

def escape_text(value):
    if value is None:
        return "-"
    return html.escape(str(value))

def image_to_base64(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_types.get(suffix, "application/octet-stream")
    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )

def render_dataset_overview(profile):
    dataset = profile["dataset"]
    return f"""
    <div class="summary-grid">

        <div class="summary-card">
            <div>数据行数</div>
            <div class="value">
                {dataset["rows"]:,}
            </div>
        </div>

        <div class="summary-card">
            <div>变量数量</div>
            <div class="value">
                {dataset["columns"]:,}
            </div>
        </div>

        <div class="summary-card">
            <div>缺失值数量</div>
            <div class="value">
                {dataset["missing_values"]:,}
            </div>
        </div>

        <div class="summary-card">
            <div>重复行数量</div>
            <div class="value">
                {dataset["duplicate_rows"]:,}
            </div>
        </div>

        <div class="summary-card">
            <div>内存占用</div>
            <div class="value">
                {dataset["memory_usage"] / 1024 / 1024:.2f}
                MB
            </div>
        </div>

    </div>
    """

def render_column_overview(profile):
    rows = []
    for column in profile["columns"]:
        rows.append(
            f"""
            <tr>

                <td>
                    {escape_text(column["name"])}
                </td>

                <td>
                    {escape_text(column["dtype"])}
                </td>

                <td>
                    {column["non_null"]:,}
                </td>

                <td>
                    {column["missing"]:,}
                </td>

                <td>
                    {column["missing_ratio"]:.2%}
                </td>

                <td>
                    {column["unique"]:,}
                </td>

            </tr>
            """
        )
    return f"""
    <table>

        <thead>

            <tr>
                <th>变量名</th>
                <th>数据类型</th>
                <th>非空数量</th>
                <th>缺失数量</th>
                <th>缺失比例</th>
                <th>唯一值数量</th>
            </tr>

        </thead>

        <tbody>

            {"".join(rows)}

        </tbody>

    </table>
    """

def render_numeric_statistics(statistics):
    numeric = statistics["numeric"]
    if not numeric:
        return "<p>数据集中没有数值变量。</p>"
    sections = []
    for column, result in numeric.items():
        quantiles = result["quantiles"]
        outliers = result["outliers"]
        normality = result["normality_test"]
        sections.append(
            f"""
            <h3>
                {escape_text(column)}
            </h3>

            <table>

                <tbody>

                    <tr>
                        <td>样本数</td>
                        <td>{result["count"]:,}</td>

                        <td>缺失值</td>
                        <td>{result["missing"]:,}</td>
                    </tr>

                    <tr>
                        <td>均值</td>
                        <td>{format_number(result["mean"])}</td>

                        <td>中位数</td>
                        <td>{format_number(result["median"])}</td>
                    </tr>

                    <tr>
                        <td>标准差</td>
                        <td>{format_number(result["std"])}</td>

                        <td>方差</td>
                        <td>{format_number(result["variance"])}</td>
                    </tr>

                    <tr>
                        <td>最小值</td>
                        <td>{format_number(result["min"])}</td>

                        <td>最大值</td>
                        <td>{format_number(result["max"])}</td>
                    </tr>

                    <tr>
                        <td>偏度</td>
                        <td>{format_number(result["skewness"])}</td>

                        <td>峰度</td>
                        <td>{format_number(result["kurtosis"])}</td>
                    </tr>

                    <tr>
                        <td>MAD</td>
                        <td>{format_number(result["mad"])}</td>

                        <td>变异系数</td>
                        <td>
                            {format_number(result["coefficient_of_variation"])}
                        </td>
                    </tr>

                </tbody>

            </table>

            <p>
                <strong>分位数：</strong>

                Q01 =
                {format_number(quantiles["q01"])},

                Q05 =
                {format_number(quantiles["q05"])},

                Q25 =
                {format_number(quantiles["q25"])},

                Q50 =
                {format_number(quantiles["q50"])},

                Q75 =
                {format_number(quantiles["q75"])},

                Q95 =
                {format_number(quantiles["q95"])},

                Q99 =
                {format_number(quantiles["q99"])}
            </p>

            <p>

                <strong>潜在异常值：</strong>

                {outliers["count"]:,}
                （{outliers["ratio"]:.2%}）

            </p>

            <p>

                <strong>正态性检验：</strong>

                {escape_text(normality["test"])}

                ，
                p =
                {format_number(normality["p_value"])}

            </p>
            """
        )
    return "".join(sections)

def render_categorical_statistics(statistics):
    categorical = statistics["categorical"]
    if not categorical:
        return "<p>数据集中没有分类变量。</p>"
    sections = []
    for column, result in categorical.items():
        frequencies = result["frequencies"]
        frequency_rows = []
        for item in frequencies[:20]:
            frequency_rows.append(
                f"""
                <tr>

                    <td>
                        {escape_text(item["category"])}
                    </td>

                    <td>
                        {item["count"]:,}
                    </td>

                    <td>
                        {item["ratio"]:.2%}
                    </td>

                </tr>
                """
            )
        sections.append(
            f"""
            <h3>
                {escape_text(column)}
            </h3>

            <p>

                唯一值数量：
                <strong>
                    {result["unique"]:,}
                </strong>

                ；
                众数：
                <strong>
                    {escape_text(result["mode"])}
                </strong>

                ；
                众数比例：
                <strong>
                    {result["mode_ratio"]:.2%}
                </strong>

                ；
                Shannon 熵：
                <strong>
                    {result["entropy"]:.4f}
                </strong>

            </p>

            <table>

                <thead>

                    <tr>
                        <th>类别</th>
                        <th>数量</th>
                        <th>比例</th>
                    </tr>

                </thead>

                <tbody>

                    {"".join(frequency_rows)}

                </tbody>

            </table>
            """
        )
    return "".join(sections)

def render_statistics(statistics):
    return (render_numeric_statistics(statistics) + render_categorical_statistics(statistics))

def render_quality(quality):
    sections = []
    missing = quality["missing"]
    missing_rows = []
    for column, result in missing["columns"].items():
        missing_rows.append(
            f"""
            <tr>

                <td>
                    {escape_text(column)}
                </td>

                <td>
                    {result["count"]:,}
                </td>

                <td>
                    {result["ratio"]:.2%}
                </td>

                <td class="{result["level"]}">
                    {result["level"]}
                </td>

            </tr>
            """
        )
    sections.append(
        f"""
        <h3>缺失值</h3>

        <p>
            总缺失值：
            <strong>
                {missing["total_missing"]:,}
            </strong>
        </p>

        <table>

            <thead>

                <tr>
                    <th>变量</th>
                    <th>缺失数量</th>
                    <th>缺失比例</th>
                    <th>状态</th>
                </tr>

            </thead>

            <tbody>
                {"".join(missing_rows)}
            </tbody>

        </table>
        """
    )
    duplicates = quality["duplicates"]
    sections.append(
        f"""
        <h3>重复数据</h3>

        <p>
            重复行：
            <strong>
                {duplicates["count"]:,}
            </strong>

            ，占总数据
            <strong>
                {duplicates["ratio"]:.2%}
            </strong>
        </p>
        """
    )
    constants = quality["constant_columns"]
    if constants["count"] > 0:
        columns = "、".join(escape_text(column) for column in constants["columns"])
        sections.append(
            f"""
            <h3>常数变量</h3>

            <p class="warning">
                发现 {constants["count"]} 个常数变量：
                {columns}
            </p>
            """
        )
    else:
        sections.append(
            """
            <h3>常数变量</h3>

            <p class="ok">
                未发现常数变量。
            </p>
            """
        )
    outliers = quality["outliers"]
    sections.append(
        """
        <h3>潜在异常值</h3>
        """
    )
    outlier_rows = []
    for column, result in outliers.items():
        outlier_rows.append(
            f"""
            <tr>

                <td>
                    {escape_text(column)}
                </td>

                <td>
                    {result["count"]:,}
                </td>

                <td>
                    {result["ratio"]:.2%}
                </td>

                <td>
                    [{format_number(result["lower_bound"])},
                    {format_number(result["upper_bound"])}]
                </td>

            </tr>
            """
        )
    sections.append(
        f"""
        <table>

            <thead>

                <tr>
                    <th>变量</th>
                    <th>异常值数量</th>
                    <th>比例</th>
                    <th>IQR范围</th>
                </tr>

            </thead>

            <tbody>

                {"".join(outlier_rows)}

            </tbody>

        </table>
        """
    )
    return "".join(sections)

def render_missingness(missingness):
    if not missingness:
        return "<p>暂无缺失值分析结果。</p>"
    summary = missingness.get("summary", {})
    variable_missingness = missingness.get("variable_missingness", [])
    missing_patterns = missingness.get("missing_patterns", [])
    pairwise_missingness = missingness.get("pairwise_missingness", [])
    html = []
    html.append("<h3>缺失值总体情况</h3>")
    html.append("""
    <table>
        <thead>
            <tr>
                <th>指标</th>
                <th>数值</th>
            </tr>
        </thead>
        <tbody>
    """)
    html.append(
        f"""
        <tr>
            <td>总缺失单元格</td>
            <td>{summary.get("total_missing", 0)}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>总体缺失率</td>
            <td>{summary.get("overall_missing_ratio", 0):.2%}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>存在缺失值的变量</td>
            <td>{summary.get("columns_with_missing", 0)}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>完全无缺失的变量</td>
            <td>{summary.get("columns_without_missing", 0)}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>完整数据行</td>
            <td>{summary.get("complete_rows", 0)}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>完整数据行比例</td>
            <td>{summary.get("complete_row_ratio", 0):.2%}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>包含缺失值的数据行</td>
            <td>{summary.get("rows_with_missing", 0)}</td>
        </tr>
        """
    )
    html.append(
        f"""
        <tr>
            <td>包含缺失值的数据行比例</td>
            <td>{summary.get("rows_with_missing_ratio", 0):.2%}</td>
        </tr>
        """
    )
    html.append("""
        </tbody>
    </table>
    """)
    html.append("<h3>变量缺失情况</h3>")
    if variable_missingness:
        html.append("""
        <table>
            <thead>
                <tr>
                    <th>变量</th>
                    <th>缺失数量</th>
                    <th>非缺失数量</th>
                    <th>缺失率</th>
                </tr>
            </thead>
            <tbody>
        """)
        for item in variable_missingness:
            html.append(
                f"""
                <tr>
                    <td>{item.get("column", "")}</td>
                    <td>{item.get("missing_count", 0)}</td>
                    <td>{item.get("non_missing_count", 0)}</td>
                    <td>{item.get("missing_ratio", 0):.2%}</td>
                </tr>
                """
            )
        html.append("""
            </tbody>
        </table>
        """)
    else:
        html.append("<p>没有检测到变量缺失信息。</p>")
    html.append("<h3>常见缺失模式</h3>")
    if missing_patterns:
        html.append("""
        <table>
            <thead>
                <tr>
                    <th>缺失变量</th>
                    <th>缺失变量数量</th>
                    <th>出现次数</th>
                    <th>占数据行比例</th>
                </tr>
            </thead>
            <tbody>
        """)
        for item in missing_patterns:
            columns = item.get("missing_columns", [])
            if columns:
                columns_text = ", ".join(str(column) for column in columns)
            else:
                columns_text = "无缺失"
            html.append(
                f"""
                <tr>
                    <td>{columns_text}</td>
                    <td>{item.get("missing_variable_count", 0)}</td>
                    <td>{item.get("missing_count", 0)}</td>
                    <td>{item.get("missing_ratio", 0):.2%}</td>
                </tr>
                """
            )
        html.append("""
            </tbody>
        </table>
        """)
    else:
        html.append("<p>没有检测到明显的缺失模式。</p>")
    html.append("<h3>变量共同缺失情况</h3>")
    if pairwise_missingness:
        html.append("""
        <table>
            <thead>
                <tr>
                    <th>变量 1</th>
                    <th>变量 2</th>
                    <th>共同缺失数量</th>
                    <th>共同缺失比例</th>
                </tr>
            </thead>
            <tbody>
        """)
        for item in pairwise_missingness:
            html.append(
                f"""
                <tr>
                    <td>{item.get("column_1", "")}</td>
                    <td>{item.get("column_2", "")}</td>
                    <td>{item.get("both_missing_count", 0)}</td>
                    <td>{item.get("both_missing_ratio", 0):.2%}</td>
                </tr>
                """
            )
        html.append("""
            </tbody>
        </table>
        """)
    else:
        html.append("<p>没有足够的变量用于共同缺失分析。</p>")
    return "\n".join(html)

def render_time_analysis(time_analysis):
    if not time_analysis:
        return "<p>暂无时间分析结果。</p>"
    if time_analysis.get("status") == "skipped":
        reason = time_analysis.get("reason", "unknown")
        return (
            "<p>时间分析已跳过。"
            f"原因：{reason}</p>"
        )
    analyses = time_analysis.get("analyses", {})
    if not analyses:
        return "<p>没有可用的时间变量。</p>"
    sections = []
    for column, result in analyses.items():
        sections.append(
            f"<h3>时间变量：{column}</h3>"
        )
        time_range = result.get("time_range", {})
        if time_range.get("status") == "ok":
            span_days = time_range.get("span_days", 0)
            sections.append(
                "<h4>时间范围</h4>"
                "<table>"
                "<tr>"
                "<th>指标</th>"
                "<th>结果</th>"
                "</tr>"
                f"<tr><td>最早时间</td>"
                f"<td>{time_range.get('min_datetime', '-')}</td></tr>"
                f"<tr><td>最晚时间</td>"
                f"<td>{time_range.get('max_datetime', '-')}</td></tr>"
                f"<tr><td>时间跨度</td>"
                f"<td>{span_days:.2f} 天</td></tr>"
                f"<tr><td>有效时间数量</td>"
                f"<td>{time_range.get('valid_count', 0)}</td></tr>"
                f"<tr><td>缺失时间数量</td>"
                f"<td>{time_range.get('missing_count', 0)}</td></tr>"
                "</table>"
            )
        intervals = result.get("time_intervals", {})
        if intervals.get("status") == "ok":
            median_minutes = (intervals.get("median_interval_seconds", 0) / 60)
            mean_minutes = (intervals.get("mean_interval_seconds", 0) / 60)
            sections.append(
                "<h4>时间间隔</h4>"
                "<table>"
                "<tr>"
                "<th>指标</th>"
                "<th>结果</th>"
                "</tr>"
                f"<tr><td>平均间隔</td>"
                f"<td>{mean_minutes:.2f} 分钟</td></tr>"
                f"<tr><td>中位数间隔</td>"
                f"<td>{median_minutes:.2f} 分钟</td></tr>"
                f"<tr><td>最小间隔</td>"
                f"<td>{intervals.get('min_interval_seconds', 0) / 60:.2f} 分钟</td></tr>"
                f"<tr><td>最大间隔</td>"
                f"<td>{intervals.get('max_interval_seconds', 0) / 60:.2f} 分钟</td></tr>"
                "</table>"
            )
            interval_rows = []
            for item in intervals.get("intervals", []):
                interval_rows.append(
                    "<tr>"
                    f"<td>{item.get('interval_minutes', 0):.2f}</td>"
                    f"<td>{item.get('count', 0)}</td>"
                    f"<td>{item.get('ratio', 0) * 100:.2f}%</td>"
                    "</tr>"
                )
            if interval_rows:
                sections.append(
                    "<h5>常见时间间隔</h5>"
                    "<table>"
                    "<tr>"
                    "<th>间隔（分钟）</th>"
                    "<th>出现次数</th>"
                    "<th>占比</th>"
                    "</tr>"
                    + "".join(interval_rows)
                    + "</table>"
                )
        hourly = result.get("hourly_distribution", {})
        if hourly.get("status") == "ok":
            rows = []
            for item in hourly.get("results", []):
                rows.append(
                    "<tr>"
                    f"<td>{item.get('hour', 0):02d}:00</td>"
                    f"<td>{item.get('count', 0)}</td>"
                    f"<td>{item.get('ratio', 0) * 100:.2f}%</td>"
                    "</tr>"
                )
            sections.append(
                "<h4>小时分布</h4>"
                "<table>"
                "<tr>"
                "<th>小时</th>"
                "<th>数据量</th>"
                "<th>占比</th>"
                "</tr>"
                + "".join(rows)
                + "</table>"
            )
        weekday = result.get("weekday_distribution", {})
        if weekday.get("status") == "ok":
            rows = []
            for item in weekday.get("results", []):
                rows.append(
                    "<tr>"
                    f"<td>{item.get('weekday', '-')}</td>"
                    f"<td>{item.get('count', 0)}</td>"
                    f"<td>{item.get('ratio', 0) * 100:.2f}%</td>"
                    "</tr>"
                )
            sections.append(
                "<h4>星期分布</h4>"
                "<table>"
                "<tr>"
                "<th>星期</th>"
                "<th>数据量</th>"
                "<th>占比</th>"
                "</tr>"
                + "".join(rows)
                + "</table>"
            )
        daily = result.get("daily_distribution", {})
        if daily.get("status") == "ok":
            daily_results = daily.get("results", [])
            if daily_results:
                rows = []
                for item in daily_results:
                    rows.append(
                        "<tr>"
                        f"<td>{item.get('date', '-')}</td>"
                        f"<td>{item.get('count', 0)}</td>"
                        "</tr>"
                    )
                if daily.get("truncated"):
                    sections.append(
                        "<p>"
                        f"日期数量共 {daily.get('unique_dates', 0)} 天，"
                        f"表格仅显示最近 "
                        f"{daily.get('displayed_results', 0)} 天。"
                        "</p>"
                    )
                sections.append("<h4>每日数据量</h4>" "<table>" "<tr>" "<th>日期</th>" "<th>数据量</th>" "</tr>" + "".join(rows) + "</table>")
    return "\n".join(sections)

def render_correlation(correlation):
    sections = []
    important_pairs = (correlation["important_pairs"])
    for method in ["pearson", "spearman", "kendall"]:
        pairs = important_pairs.get(method, [])
        sections.append(
            f"""
            <h3>
                {method.capitalize()}
            </h3>
            """
        )
        if not pairs:
            sections.append("<p>没有发现超过阈值的强相关变量对。</p>")
            continue
        for pair in pairs[:30]:
            direction_text = ("正相关" if pair["direction"] == "positive" else "负相关")
            sections.append(
                f"""
                <div class="correlation-pair">

                    <strong>
                        {escape_text(pair["variable_1"])}
                    </strong>

                    ↔

                    <strong>
                        {escape_text(pair["variable_2"])}
                    </strong>

                    <br>

                    相关系数：
                    <strong>
                        {format_number(pair["correlation"])}
                    </strong>

                    ，
                    方向：
                    {direction_text}

                    ，
                    p-value：
                    {format_number(pair["p_value"])}

                    ，
                    样本量：
                    {pair["sample_size"]:,}

                </div>
                """
            )
    return "".join(sections)

def render_visualization(visualizations):
    if not visualizations:
        return "<p>没有生成可视化结果。</p>"
    sections = []
    visualization_groups = [
        ("数值变量直方图", "histograms"),
        ("数值变量箱线图", "boxplots"),
        ("分类变量分布", "categorical_bars"),
        ("变量关系散点图", "scatter_plots")
    ]
    for title, key in visualization_groups:
        images = visualizations.get(key, [])
        if not images:
            continue
        image_html = []
        for image_path in images:
            image_data = image_to_base64(image_path)
            if image_data is None:
                continue
            image_html.append(
                f"""
                <div class="plot-card">

                    <img
                        src="{image_data}"
                        alt="{escape_text(title)}"
                    >

                </div>
                """
            )
        if not image_html:
            continue
        sections.append(
            f"""
            <h3>
                {escape_text(title)}
            </h3>

            <div class="plot-grid">
                {"".join(image_html)}
            </div>
            """
        )
    time_visualization_groups = [("小时分布", "time_hourly"), ("星期分布", "time_weekday"), ("每日数据量趋势", "time_daily")]
    for title, key in time_visualization_groups:
        images = visualizations.get(key, [])
        if not images:
            continue
        image_html = []
        for image_path in images:
            image_data = image_to_base64(image_path)
            if image_data is None:
                continue
            image_html.append(
                f"""
                <div class="plot-card">

                    <img
                        src="{image_data}"
                        alt="{escape_text(title)}"
                    >

                </div>
                """
            )
        if not image_html:
            continue
        sections.append(
            f"""
            <h3>
                {escape_text(title)}
            </h3>

            <div class="plot-grid">
                {"".join(image_html)}
            </div>
            """
        )
    time_series_images = visualizations.get("time_series", [])
    if time_series_images:
        image_html = []
        for image_path in time_series_images:
            image_data = image_to_base64(image_path)
            if image_data is None:
                continue
            image_html.append(
                f"""
                <div class="plot-card">

                    <img
                        src="{image_data}"
                        alt="时间序列图"
                    >

                </div>
                """
            )
        if image_html:
            sections.append(
                """
                <h3>
                    时间序列
                </h3>

                <div class="plot-grid">
                    """
                + "".join(image_html)
                + """
                </div>
                """
            )
    heatmap = visualizations.get("correlation_heatmap")
    if heatmap:
        image_data = image_to_base64(heatmap)
        if image_data is not None:
            sections.append(
                """
                <h3>相关性热力图</h3>

                <div class="plot-grid">

                    <div class="plot-card">

                        <img
                            src="{image_data}"
                            alt="相关性热力图"
                        >

                    </div>

                </div>
                """.replace("{image_data}", image_data)
            )
    if not sections:
        return "<p>没有生成可用的可视化结果。</p>"
    return "".join(sections)

def generate_report(
    profile,
    statistics,
    quality,
    missingness,
    time_analysis,
    correlation,
    visualizations,
    output_path=DEFAULT_REPORT_PATH,
    title="自动化数据分析报告"
):
    """生成完整的 HTML 分析报告。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as file:
        template = file.read()
    replacements = {
        "{{TITLE}}":
            escape_text(title),
        "{{DATASET_OVERVIEW}}":
            render_dataset_overview(profile),
        "{{COLUMN_OVERVIEW}}":
            render_column_overview(profile),
        "{{STATISTICS}}":
            render_statistics(statistics),
        "{{QUALITY}}":
            render_quality(quality),
        "{{MISSINGNESS}}":
            render_missingness(missingness),
        "{{TIME_ANALYSIS}}":
            render_time_analysis(time_analysis),
        "{{CORRELATION}}":
            render_correlation(correlation),
        "{{VISUALIZATION}}":
            render_visualization(visualizations)
    }
    html_content = template
    for placeholder, content in replacements.items():
        html_content = html_content.replace(placeholder, content)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    return output_path
