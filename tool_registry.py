"""定义面向后续 Agent 的工具注册信息。"""
from __future__ import annotations
from importlib import import_module
from typing import Any, Callable
_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "profile": {
        "name": "profile",
        "function": "tools.data_profiler:analyze_profile",
        "category": "profiling",
        "description": ("Detect semantic variable types and build a dataset profile."),
        "when_to_use": (
            "Use first when the dataset has not yet been profiled or when "
            "the Agent needs variable-level semantic information."
        ),
        "inputs": ["dataframe"],
        "outputs": ["profile", "variable_types"],
        "depends_on": [],
        "changes_data": False,
        "large_data": "sample_based",
    },
    "preprocess": {
        "name": "preprocess",
        "function": "tools.data_preprocessor:preprocess_dataframe",
        "category": "preprocessing",
        "description": ("Clean and convert variables according to detected semantic types."),
        "when_to_use": (
            "Use after profiling and before statistical or time-based analysis "
            "when the raw DataFrame may contain numeric, date, or time values "
            "stored as strings."
        ),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["dataframe", "transformations"],
        "depends_on": ["profile"],
        "changes_data": True,
        "large_data": "full_dataframe",
    },
    "statistics": {
        "name": "statistics",
        "function": "tools.statistics:analyze_statistics",
        "category": "descriptive_statistics",
        "description": ("Calculate descriptive statistics for numeric and categorical variables."),
        "when_to_use": (
            "Use when the Agent needs distributions, central tendency, "
            "dispersion, quantiles, outliers, or categorical frequencies."
        ),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["numeric", "categorical"],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "full_dataframe",
    },
    "quality": {
        "name": "quality",
        "function": "tools.quality:analyze_quality",
        "category": "data_quality",
        "description": ("Check missing values, duplicates, constants, cardinality, " "outliers, skewness, and type issues."),
        "when_to_use": ("Use when the Agent needs to assess whether the dataset is " "clean and suitable for downstream analysis."),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["summary", "missing", "duplicates", "constant_columns", "high_cardinality", "outliers", "skewness", "type_issues",],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "full_dataframe",
    },
    "missingness": {
        "name": "missingness",
        "function": "tools.missingness:analyze_missingness",
        "category": "missing_data",
        "description": ("Analyze missing-value totals, variable-level missingness, " "patterns, and pairwise missingness."),
        "when_to_use": (
            "Use when missing data is important to the analysis or when "
            "the Agent needs to understand incomplete observations."
        ),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["summary", "variable_missingness", "missing_patterns", "pairwise_missingness",],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "full_dataframe",
    },
    "time_analysis": {
        "name": "time_analysis",
        "function": "tools.time_analysis:analyze_time_analysis",
        "category": "temporal_analysis",
        "description": ("Analyze datetime ranges, intervals, hourly distributions, " "daily distributions, and weekday distributions."),
        "when_to_use": ("Use when one or more datetime variables are available and " "the Agent needs calendar-based patterns."),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["datetime_columns", "analyses",],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "full_dataframe",
    },
    "time_series": {
        "name": "time_series",
        "function": "tools.time_series:analyze_time_series",
        "category": "temporal_analysis",
        "description": ("Analyze time-series frequency, trends, aggregation, and anomalies."),
        "when_to_use": (
            "Use when a datetime variable and numeric variables are available "
            "and the Agent needs ordered temporal behavior."
        ),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["datetime_column", "time_range", "frequency", "original_series", "aggregation",],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "frequency_dependent",
    },
    "correlation": {
        "name": "correlation",
        "function": "tools.correlation:analyze_correlation",
        "category": "relationship_analysis",
        "description": ("Calculate Pearson, Spearman, and conditionally Kendall " "correlations and identify important numeric pairs."),
        "when_to_use": (
            "Use when the Agent needs relationships among numeric variables "
            "or wants to identify strongly associated pairs."
        ),
        "inputs": ["dataframe", "variable_types"],
        "outputs": ["pearson", "spearman", "kendall", "important_pairs",],
        "depends_on": ["profile", "preprocess"],
        "changes_data": False,
        "large_data": "kendall_limited",
    },
    "visualization": {
        "name": "visualization",
        "function": "tools.visualization:create_all_visualizations",
        "category": "visualization",
        "description": (
            "Generate static charts including distributions, boxplots, "
            "categorical bars, scatter plots, correlation heatmaps, "
            "calendar distributions, and time-series charts."
        ),
        "when_to_use": ("Use after the relevant analysis results exist and the Agent " "needs visual evidence or report-ready figures."),
        "inputs": ["dataframe", "output_dir", "correlation_result", "variable_types", "time_analysis", "time_series_analysis",],
        "outputs": ["figure_paths", "chart_groups"],
        "depends_on": ["profile", "preprocess", "statistics", "correlation", "time_analysis", "time_series",],
        "changes_data": False,
        "large_data": "sampled_for_plots",
    },
}

def get_tool_registry() -> dict[str, dict[str, Any]]:
    """获取所有可供 Agent 使用的工具说明。"""
    return {name: dict(spec) for name, spec in _TOOL_SPECS.items()}

def list_tools() -> list[str]:
    return list(_TOOL_SPECS.keys())

def get_tool_spec(name: str) -> dict[str, Any]:
    """获取指定工具的说明。"""
    if name not in _TOOL_SPECS:
        raise KeyError(f"Unknown analysis tool: {name}")
    return dict(_TOOL_SPECS[name])

def resolve_tool(name: str) -> Callable[..., Any]:
    """将工具名称解析为对应的 Python 函数。"""
    spec = get_tool_spec(name)
    module_name, function_name = spec["function"].split(":", 1)
    module = import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(
            f"Registered tool is not callable: {spec['function']}"
        )
    return function

def invoke_tool(name: str, *, dataframe: Any = None, variable_types: Any = None, **kwargs: Any,) -> Any:
    """调用已注册的分析工具。"""
    function = resolve_tool(name)
    call_kwargs = dict(kwargs)
    if dataframe is not None:
        call_kwargs.setdefault("df", dataframe)
    if variable_types is not None:
        call_kwargs.setdefault("variable_types", variable_types,)
    return function(**call_kwargs)
