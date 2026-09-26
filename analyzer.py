from pathlib import Path
from tools.data_loader import load_and_prepare_csv
from tools.data_profiler import (analyze_profile, apply_variable_type_overrides,)
from tools.data_preprocessor import (preprocess_dataframe, summarize_preprocessing, update_profile_after_preprocessing)
from tools.statistics import analyze_statistics
from tools.quality import analyze_quality
from tools.missingness import analyze_missingness
from tools.correlation import analyze_correlation
from tools.visualization import (create_all_visualizations)
from tools.time_analysis import (analyze_time_analysis)
from tools.time_series import (analyze_time_series)
from report.generator import generate_report
from report.dashboard_generator import generate_dashboard
from analysis_result import (build_analysis_result)
PROJECT_ROOT = Path(__file__).resolve().parent

def analyze_csv(input_file, output_dir="output", progress_callback=None, variable_type_overrides=None):
    """执行完整的 CSV 分析流程。"""
    def report_progress(message, percentage):
        if progress_callback is not None:
            progress_callback(message, percentage)
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    report_progress("正在检查输入文件...", 2)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{input_file}"
        )
    if not input_file.is_file():
        raise ValueError(
            f"Input path is not a file: "
            f"{input_file}"
        )
    if input_file.suffix.lower() != ".csv":
        raise ValueError(
            f"Input file is not a CSV file: "
            f"{input_file}"
        )
    report_progress("正在准备输出目录...", 4)
    figure_dir = (output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_progress("正在读取 CSV 数据...", 8)
    print(
        f"Input file: {input_file}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print()
    print("Loading CSV...")
    df, loading_info = (load_and_prepare_csv(input_file))
    print(
        f"Loaded dataset: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )
    report_progress(
        f"CSV 读取完成："
        f"{df.shape[0]:,} 行 × "
        f"{df.shape[1]} 列",
        15
    )
    report_progress("正在识别变量类型...", 18)
    print("Analyzing variable types...")
    profile = analyze_profile(df)
    # 有用户类型覆盖时使用覆盖结果，否则保持自动识别。
    profile = apply_variable_type_overrides(profile, variable_type_overrides)
    variable_types = (profile["variable_types"])
    report_progress("变量类型识别完成", 23)
    report_progress("正在进行数据预处理...", 26)
    print("Preprocessing data...")
    df, transformations = (preprocess_dataframe(df, variable_types))
    preprocessing_summary = (summarize_preprocessing(transformations))
    print("Preprocessing completed:")
    print(preprocessing_summary)
    report_progress("数据预处理完成", 32)
    report_progress("正在更新变量信息...", 34)
    profile = (update_profile_after_preprocessing(profile, df))
    report_progress("变量信息更新完成", 36)
    report_progress("正在进行统计分析...", 39)
    print("Running statistical analysis...")
    statistics = analyze_statistics(df, variable_types)
    report_progress("统计分析完成", 45)
    report_progress("正在进行数据质量分析...", 47)
    print("Running data quality analysis...")
    quality = analyze_quality(df, variable_types)
    report_progress("数据质量分析完成", 53)
    report_progress("正在分析缺失值...", 55)
    print("Running missingness analysis...")
    missingness = analyze_missingness(df, variable_types)
    print("Missingness analysis completed:")
    print("Total missing values:", missingness["summary"]["total_missing"])
    print("Complete rows:", missingness["summary"]["complete_rows"])
    print("Variables with missing values:", missingness["summary"]["columns_with_missing"])
    report_progress("缺失值分析完成", 60)
    report_progress("正在进行时间变量分析...", 62)
    print("Running time analysis...")
    time_analysis = (analyze_time_analysis(df, variable_types))
    print("Time analysis completed:")
    print("Datetime columns:", time_analysis.get("datetime_columns", []))
    report_progress("时间变量分析完成", 67)
    report_progress("正在进行时间序列分析...", 69)
    print("Running time series analysis...")
    time_series_analysis = (analyze_time_series(df, variable_types))
    print("Time series analysis completed:")
    if (time_series_analysis.get("status") == "ok"):
        print("Datetime column:", time_series_analysis.get("datetime_column"))
        print("Numeric variables:", len(time_series_analysis.get("numeric_columns", [])))
        print("Detected frequency:", time_series_analysis.get("frequency", {}).get("detected"))
    else:
        print("Skipped:", time_series_analysis.get("reason"))
    report_progress("时间序列分析完成", 74)
    report_progress("正在进行相关性分析...", 76)
    print("Running correlation analysis...")
    correlation = analyze_correlation(df, variable_types)
    report_progress("相关性分析完成", 82)
    report_progress("正在生成数据可视化图表...", 84)
    print("Creating visualizations...")
    visualizations = (
        create_all_visualizations(
            df,
            output_dir=figure_dir,
            correlation_result=correlation,
            correlation_method="spearman",
            variable_types=variable_types,
            time_analysis=time_analysis,
            time_series_analysis=(time_series_analysis)
        )
    )
    report_progress("数据可视化生成完成", 90)
    report_progress("正在生成静态 HTML 分析报告...", 92)
    print("Generating HTML report...")
    report_path = (output_dir / "report.html")
    generate_report(
        profile=profile,
        statistics=statistics,
        quality=quality,
        missingness=missingness,
        time_analysis=time_analysis,
        correlation=correlation,
        visualizations=visualizations,
        output_path=report_path
    )
    report_progress("静态分析报告生成完成", 95)
    report_progress("正在生成交互式 Dashboard...", 96)
    print("Generating interactive dashboard...")
    dashboard_path = (output_dir / "dashboard.html")
    generate_dashboard(
        df=df,
        profile=profile,
        statistics=statistics,
        quality=quality,
        missingness=missingness,
        time_analysis=time_analysis,
        time_series=(time_series_analysis),
        correlation=correlation,
        visualizations=visualizations,
        output_path=dashboard_path,
        template_path=(PROJECT_ROOT / "report" / "dashboard_template.html"),
        title="交互式数据分析报告"
    )
    report_progress("交互式 Dashboard 生成完成", 99)
    result = build_analysis_result(
        input_file=input_file,
        output_dir=output_dir,
        dataframe=df,
        loading_info=loading_info,
        profile=profile,
        variable_types=variable_types,
        transformations=transformations,
        preprocessing_summary=(preprocessing_summary),
        statistics=statistics,
        quality=quality,
        missingness=missingness,
        time_analysis=time_analysis,
        time_series=(time_series_analysis),
        correlation=correlation,
        visualizations=visualizations,
        report_path=report_path,
        dashboard_path=dashboard_path,
        variable_type_overrides=(variable_type_overrides or {}),
    )
    report_progress("分析全部完成", 100)
    print()
    print("=" * 60)
    print("Analysis completed successfully.")
    print(
        f"Report: {report_path}"
    )
    print(
        f"Dashboard: {dashboard_path}"
    )
    print("=" * 60)
    return result
