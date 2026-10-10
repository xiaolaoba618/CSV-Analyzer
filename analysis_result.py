from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
SCHEMA_VERSION = "1.0"

class AnalysisResult(dict):
    """一次完整 CSV 分析的结果容器，兼容旧接口并提供统一结果结构。"""
    schema_version = SCHEMA_VERSION
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    @property
    def analysis(self) -> dict[str, Any]:
        return self["analysis"]
    @property
    def outputs(self) -> dict[str, Any]:
        return self["outputs"]
    @property
    def dataframe(self) -> Any:
        return self.get("dataframe")

def build_analysis_result(
    *,
    input_file: str | Path,
    output_dir: str | Path,
    dataframe: Any,
    loading_info: Any,
    profile: Any,
    variable_types: Any,
    transformations: Any,
    preprocessing_summary: Any,
    statistics: Any,
    quality: Any,
    missingness: Any,
    time_analysis: Any,
    time_series: Any,
    correlation: Any,
    visualizations: Any,
    report_path: str | Path,
    dashboard_path: str | Path,
    variable_type_overrides: Any = None,
) -> AnalysisResult:
    """构建统一的分析结果对象。"""
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    dashboard_path = Path(dashboard_path)
    result = AnalysisResult(
        schema_version=SCHEMA_VERSION,
        status="completed",
        metadata={"input_file": str(input_file), "output_dir": str(output_dir),},
        dataset={
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "column_names": [str(column) for column in dataframe.columns],
        },
        variables={"variable_types": variable_types, "user_overrides": variable_type_overrides or {},},
        analysis={
            "profile": profile,
            "preprocessing": {"transformations": transformations, "summary": preprocessing_summary,},
            "statistics": statistics,
            "quality": quality,
            "missingness": missingness,
            "time": time_analysis,
            "time_series": time_series,
            "correlation": correlation,
            "visualizations": visualizations,
        },
        outputs={"report": str(report_path), "dashboard": str(dashboard_path), "figure_directory": str(output_dir / "figures"),},
        # 保留原始 DataFrame，便于 Python 调用。
        # 对外结果中不直接暴露原始 DataFrame。
        dataframe=dataframe,
        input_file=str(input_file),
        output_dir=str(output_dir),
        loading_info=loading_info,
        profile=profile,
        variable_types=variable_types,
        transformations=transformations,
        preprocessing_summary=preprocessing_summary,
        statistics=statistics,
        quality=quality,
        missingness=missingness,
        time_analysis=time_analysis,
        time_series=time_series,
        correlation=correlation,
        visualizations=visualizations,
        report_path=str(report_path),
        dashboard_path=str(dashboard_path),
    )
    return result

def _to_agent_safe(value: Any) -> Any:
    """递归转换常见对象，生成可供 Agent 使用的数据结构。"""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return {
                "type": "dataframe",
                "rows": int(value.shape[0]),
                "columns": int(value.shape[1]),
                "column_names": [str(column) for column in value.columns],
                "note": ("Raw dataframe omitted " "from Agent payload."),
            }
        if isinstance(value, pd.Series):
            return {
                "type": "series",
                "length": int(len(value)),
                "name": (None if value.name is None else str(value.name)),
                "note": ("Raw series omitted " "from Agent payload."),
            }
        if isinstance(value, (pd.Timestamp, pd.Timedelta,)):
            return str(value)
        if (not isinstance(value, (dict, list, tuple, set,)) and pd.isna(value)):
            return None
    if isinstance(value, Mapping):
        return {str(key): _to_agent_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set,)):
        return [_to_agent_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_agent_safe(value.item())
        except (TypeError, ValueError,):
            pass
    if isinstance(value, (str, int, float, bool,)):
        return value
    if value is None:
        return None
    return str(value)

def get_agent_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """生成供后续 Agent 使用的精简结果。"""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping " "or AnalysisResult")
    agent_payload = {
        "schema_version": result.get("schema_version", SCHEMA_VERSION),
        "status": result.get("status"),
        "metadata": result.get("metadata", {}),
        "dataset": result.get("dataset", {}),
        "variables": result.get("variables", {}),
        "analysis": result.get("analysis", {}),
        "outputs": result.get("outputs", {}),
    }
    if "loading_info" in result:
        agent_payload["loading_info"] = (result["loading_info"])
    try:
        from tool_registry import get_tool_registry
        agent_payload["available_tools"] = (get_tool_registry())
    except ImportError:
        agent_payload["available_tools"] = {}
    return _to_agent_safe(deepcopy(agent_payload))
