from pathlib import Path
import base64
import json
import html
from datetime import date, datetime
import numpy as np
import pandas as pd
MAX_CATEGORIES = 15
MAX_HEATMAP_COLUMNS = 30
MAX_SCATTER_POINTS = 5000
MAX_SCATTER_VARIABLES = 10
MAX_HISTOGRAM_POINTS = 10000
MAX_BOX_OUTLIERS = 200
MAX_TIME_POINTS = 5000
MAX_NUMERIC_POINTS = 5000
MAX_IDENTIFIER_VALUES = 300000
MAX_IDENTIFIER_COMBINATION_VALUES = 300000
MAX_IDENTIFIER_COMBINATION_FIELDS = 4
MAX_IDENTIFIER_CHART_ROWS = 2000000

def _make_json_safe(value):
    """将 Python、NumPy 和 pandas 对象转换为可安全序列化的值。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [{str(key): _make_json_safe(item) for key, item in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, np.ndarray):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            if missing:
                return None
    except (TypeError, ValueError):
        pass
    return str(value)

def _to_json(data):
    """转换为紧凑 JSON，并防止原始文本破坏 HTML 脚本节点。"""
    safe_data = _make_json_safe(data)
    json_text = json.dumps(safe_data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # 防止数据内容破坏 JSON script 节点。
    # 避免数据中的 </script> 终止 JSON 节点。
    json_text = (
        json_text
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return json_text

def _image_to_base64(image_path):
    if image_path is None:
        return None
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp"}
    mime_type = mime_types.get(suffix)
    if mime_type is None:
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return (
            f"data:{mime_type};base64,{encoded}"
        )
    except OSError:
        return None

def _build_image_data(visualizations):
    result = {}
    if not visualizations:
        return result
    if not isinstance(visualizations, dict):
        return result
    for category, paths in visualizations.items():
        if not isinstance(paths, (list, tuple)):
            continue
        images = []
        for path in paths:
            image_data = _image_to_base64(path)
            if image_data is None:
                continue
            images.append({"path": str(path), "data": image_data})
        if images:
            result[str(category)] = images
    return result

def _prepare_dataset_summary(df):
    rows, columns = df.shape
    total_cells = (rows * columns)
    if total_cells > 0:
        missing_values = int(df.isna().sum().sum())
        missing_rate = (missing_values / total_cells)
    else:
        missing_values = 0
        missing_rate = 0.0
    duplicate_rows = int(df.duplicated().sum())
    complete_rows = int(df.notna().all(axis=1).sum())
    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_values":
            missing_values,
        "missing_rate":
            float(missing_rate),
        "duplicate_rows":
            duplicate_rows,
        "complete_rows":
            complete_rows
    }

def _prepare_variable_summary(df, variable_types):
    variables = []
    for column in df.columns:
        info = variable_types.get(column, {})
        detected_type = info.get("detected_type", str(df[column].dtype))
        series = df[column]
        try:
            missing = int(series.isna().sum())
        except Exception:
            missing = 0
        try:
            missing_rate = float(series.isna().mean())
        except Exception:
            missing_rate = 0.0
        try:
            unique = int(series.nunique(dropna=True))
        except Exception:
            unique = 0
        variables.append(
            {
                "name": str(column),
                "detected_type":
                    str(detected_type),
                "dtype":
                    str(series.dtype),
                "missing":
                    missing,
                "missing_rate":
                    missing_rate,
                "unique":
                    unique
            }
        )
    return variables

def _prepare_numeric_data(df, variable_types, max_points=MAX_NUMERIC_POINTS):
    numeric_columns = [
        column
        for column, info
        in variable_types.items()
        if (info.get("detected_type") == "numeric" and column in df.columns)
    ]
    if not numeric_columns:
        return {"columns": [], "rows": []}
    categorical_columns = _get_categorical_columns(df, variable_types)
    # 保持采样行的分类字段与数值字段对应。
    columns_to_use = list(dict.fromkeys(numeric_columns + categorical_columns))
    data = df[columns_to_use].copy()
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)
    data = data.replace([np.inf, -np.inf], np.nan)
    rows = []
    for _, row in data.iterrows():
        item = {}
        for column in numeric_columns:
            value = row[column]
            if pd.isna(value):
                item[str(column)] = None
            else:
                try:
                    item[str(column)] = float(value)
                except (TypeError, ValueError):
                    item[str(column)] = None
        for column in categorical_columns:
            value = row[column]
            item[str(column)] = ("缺失" if pd.isna(value) else str(value))
        rows.append(item)
    return {"columns": [str(column) for column in numeric_columns], "rows": rows}

def _get_identifier_columns(df, variable_types, profile=None):
    columns = []
    for column, info in (variable_types or {}).items():
        if info.get("detected_type") == "identifier" and column in df.columns:
            columns.append(column)
    for info in (profile or {}).get("columns", []):
        column = info.get("name")
        if (column in df.columns and info.get("detected_type") == "identifier" and column not in columns):
            columns.append(column)
    for column, info in (variable_types or {}).items():
        if (info.get("type") == "identifier" and column in df.columns and column not in columns):
            columns.append(column)
    return columns

def _prepare_identifier_data(df, variable_types, profile=None, max_values=MAX_IDENTIFIER_VALUES):
    result = {}
    identifier_columns = _get_identifier_columns(df, variable_types, profile)
    for column in identifier_columns:
        try:
            counts = (df[column].astype("string").fillna("缺失").value_counts())
            if len(counts) > max_values:
                counts = counts.head(max_values)
            result[str(column)] = {
                "values": [str(value) for value in counts.index],
                "counts": [int(value) for value in counts.values],
                "total_unique": int(df[column].nunique(dropna=False))
            }
        except Exception:
            continue
    return result

def _prepare_identifier_combinations(
    df,
    variable_types,
    profile=None,
    max_values=MAX_IDENTIFIER_COMBINATION_VALUES,
    max_fields=MAX_IDENTIFIER_COMBINATION_FIELDS
):
    identifier_columns = _get_identifier_columns(df, variable_types, profile)
    if len(identifier_columns) < 2:
        return {}
    combinations = {}
    # 限制组合数量，避免特殊数据导致 Dashboard 数据量膨胀。
    from itertools import combinations as iter_combinations
    max_combination_size = min(max_fields, len(identifier_columns))
    for size in range(2, max_combination_size + 1):
        for selected in iter_combinations(identifier_columns, size):
            if len(combinations) >= 30:
                return combinations
            try:
                frame = df[list(selected)].copy()
                for column in selected:
                    frame[column] = frame[column].astype("string").fillna("缺失")
                counts = frame.value_counts(sort=True)
                if len(counts) > max_values:
                    counts = counts.head(max_values)
                rows = []
                for values, count in counts.items():
                    if not isinstance(values, tuple):
                        values = (values,)
                    rows.append({"values": [str(value) for value in values], "count": int(count)})
                combinations["||".join(str(column) for column in selected)] = {
                    "columns": [str(column) for column in selected],
                    "rows": rows,
                    "total_unique": int(frame.drop_duplicates().shape[0])
                }
            except Exception:
                continue
    return combinations

def _prepare_identifier_chart_data(df, variable_types, profile=None, max_rows=MAX_IDENTIFIER_CHART_ROWS):
    """准备供标识符查询联动图表的紧凑数据。"""
    identifier_columns = _get_identifier_columns(df, variable_types, profile)
    numeric_columns = [column for column, info in variable_types.items() if info.get("detected_type") == "numeric" and column in df.columns]
    if not identifier_columns or not numeric_columns or df.empty:
        return {"available": False, "reason": "no_identifier_or_numeric_data"}
    truncated = len(df) > max_rows
    if truncated:
        # 交互图表仅保留确定性的采样子集。
        data = df.iloc[:max_rows].copy()
    else:
        data = df.copy()
    time_index_columns = [
        column for column, info in variable_types.items()
        if info.get("detected_type") == "time_index" and column in data.columns
    ]
    datetime_columns = [
        column for column, info in variable_types.items()
        if info.get("detected_type") == "datetime" and column in data.columns
    ]
    categorical_columns = _get_categorical_columns(data, variable_types)
    time_type = None
    time_column = None
    if time_index_columns:
        time_type = "time_index"
        time_column = time_index_columns[0]
    elif datetime_columns:
        time_type = "datetime"
        time_column = "DATETIME" if "DATETIME" in data.columns else datetime_columns[0]
    result = {
        "available": True,
        "row_count": int(len(data)),
        "source_row_count": int(len(df)),
        "truncated": bool(truncated),
        "identifier_columns": {},
        "numeric_columns": [str(column) for column in numeric_columns],
        "numeric": {},
        "categorical_columns": [str(column) for column in categorical_columns],
        "categorical": {},
        "time_type": time_type,
        "time_column": str(time_column) if time_column is not None else None,
        "time_values": []
    }
    # 标识符使用整数编码，减少重复存储长字符串。
    for column in identifier_columns:
        try:
            codes, uniques = pd.factorize(data[column].astype("string").fillna("缺失"), sort=False)
            result["identifier_columns"][str(column)] = {
                "values": [str(value) for value in uniques],
                "codes": [int(code) for code in codes]
            }
        except Exception:
            continue
    for column in numeric_columns:
        try:
            series = pd.to_numeric(data[column], errors="coerce")
            result["numeric"][str(column)] = [None if pd.isna(value) else float(value) for value in series]
        except Exception:
            result["numeric"][str(column)] = [None] * len(data)
    # 分类值同样编码，便于复用筛选逻辑。
    for column in categorical_columns:
        try:
            codes, uniques = pd.factorize(data[column].astype("string").fillna("缺失"), sort=False)
            result["categorical"][str(column)] = {"values": [str(value) for value in uniques], "codes": [int(code) for code in codes]}
        except Exception:
            continue
    if time_column is not None:
        try:
            if time_type == "time_index":
                series = pd.to_numeric(data[time_column], errors="coerce")
                result["time_values"] = [None if pd.isna(value) else float(value) for value in series]
            else:
                series = pd.to_datetime(data[time_column], errors="coerce")
                result["time_values"] = [None if pd.isna(value) else int(value.value // 1_000_000) for value in series]
        except Exception:
            result["time_values"] = [None] * len(data)
    return result

def _prepare_categorical_data(df, variable_types, max_categories=20):
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    identifier_columns = set(_get_identifier_columns(df, variable_types))
    categorical_columns = [
        column
        for column, info
        in variable_types.items()
        if (info.get("detected_type") in categorical_types and column in df.columns and column not in identifier_columns)
    ]
    result = {}
    for column in categorical_columns:
        try:
            counts = (df[column].astype("string").fillna("缺失").value_counts().head(max_categories))
            result[str(column)] = {"categories": [str(value) for value in counts.index], "counts": [int(value) for value in counts.values]}
        except Exception:
            continue
    return result

def _prepare_time_data(df, variable_types, max_points=MAX_TIME_POINTS):
    datetime_columns = [
        column
        for column, info in variable_types.items()
        if (info.get("detected_type") == "datetime" and column in df.columns)
    ]
    time_index_columns = [
        column
        for column, info in variable_types.items()
        if (info.get("detected_type") == "time_index" and column in df.columns)
    ]
    numeric_columns = [
        column
        for column, info in variable_types.items()
        if (info.get("detected_type") == "numeric" and column in df.columns)
    ]
    if not datetime_columns and time_index_columns:
        time_column = time_index_columns[0]
        numeric_columns = [column for column in numeric_columns if column != time_column]
        if not numeric_columns:
            return {
                "available": False,
                "time_type": "time_index",
                "time_column": str(time_column),
                "datetime_column": None,
                "columns": [],
                "rows": []
            }
        categorical_columns = _get_categorical_columns(df, variable_types)
        columns_to_use = list(dict.fromkeys([time_column] + numeric_columns + categorical_columns))
        data = df[columns_to_use].copy()
        data[time_column] = pd.to_numeric(data[time_column], errors="coerce")
        data = data.dropna(subset=[time_column])
        if data.empty:
            return {
                "available": False,
                "time_type": "time_index",
                "time_column": str(time_column),
                "datetime_column": None,
                "columns": [str(column) for column in numeric_columns],
                "rows": []
            }
        data = data.sort_values(time_column)
        if len(data) > max_points:
            index = np.linspace(0, len(data) - 1, max_points, dtype=int)
            data = data.iloc[index]
        rows = []
        for _, row in data.iterrows():
            item = {"time_index": float(row[time_column])}
            for column in numeric_columns:
                value = row[column]
                item[str(column)] = (None if pd.isna(value) else float(value))
            for column in categorical_columns:
                value = row[column]
                item[str(column)] = ("缺失" if pd.isna(value) else str(value))
            rows.append(item)
        return {
            "available": True,
            "time_type": "time_index",
            "time_column": str(time_column),
            "datetime_column": None,
            "columns": [str(column) for column in numeric_columns],
            "rows": rows
        }
    if not datetime_columns:
        return {"available": False, "time_type": None, "time_column": None, "datetime_column": None, "columns": [], "rows": []}
    if not numeric_columns:
        return {
            "available": False,
            "time_type": "datetime",
            "time_column": str(datetime_columns[0]),
            "datetime_column": str(datetime_columns[0]),
            "columns": [],
            "rows": []
        }
    if "DATETIME" in datetime_columns:
        datetime_column = "DATETIME"
    else:
        datetime_column = datetime_columns[0]
    categorical_columns = _get_categorical_columns(df, variable_types)
    columns_to_use = list(dict.fromkeys([datetime_column] + numeric_columns + categorical_columns))
    data = df[columns_to_use].copy()
    data[datetime_column] = pd.to_datetime(data[datetime_column], errors="coerce")
    data = data.dropna(subset=[datetime_column])
    if data.empty:
        return {
            "available": False,
            "time_type": "datetime",
            "time_column": str(datetime_column),
            "datetime_column": str(datetime_column),
            "columns": [str(column) for column in numeric_columns],
            "rows": []
        }
    data = data.sort_values(datetime_column)
    if len(data) > max_points:
        index = np.linspace(0, len(data) - 1, max_points, dtype=int)
        data = data.iloc[index]
    rows = []
    for _, row in data.iterrows():
        item = {"datetime": row[datetime_column].isoformat()}
        for column in numeric_columns:
            value = row[column]
            if pd.isna(value):
                item[str(column)] = None
            else:
                try:
                    item[str(column)] = float(value)
                except (TypeError, ValueError):
                    item[str(column)] = None
        for column in categorical_columns:
            value = row[column]
            item[str(column)] = ("缺失" if pd.isna(value) else str(value))
        rows.append(item)
    return {
        "available": True,
        "time_type": "datetime",
        "time_column": str(datetime_column),
        "datetime_column": str(datetime_column),
        "columns": [str(column) for column in numeric_columns],
        "rows": rows
    }

def _get_numeric_columns(df, variable_types):
    return [column for column, info in variable_types.items() if (info.get("detected_type") == "numeric" and column in df.columns)]

def _get_categorical_columns(df, variable_types):
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    return [column for column, info in variable_types.items() if (info.get("detected_type") in categorical_types and column in df.columns)]

def _prepare_histogram_data(df, numeric_columns, bins=30, max_points=MAX_HISTOGRAM_POINTS):
    result = {}
    for column in numeric_columns:
        try:
            series = pd.to_numeric(df[column], errors="coerce")
            series = (series.replace([np.inf, -np.inf], np.nan).dropna())
            if len(series) == 0:
                continue
            if len(series) > max_points:
                series = series.sample(n=max_points, random_state=42)
            if series.nunique() <= 1:
                continue
            counts, edges = np.histogram(series.to_numpy(), bins=bins)
            labels = [
                (
                    f"{edges[i]:.2f} - "
                    f"{edges[i + 1]:.2f}"
                )
                for i
                in range(len(edges) - 1)
            ]
            result[str(column)] = {
                "labels": labels,
                "counts": [int(value) for value in counts],
                "mean": float(series.mean()),
                "median": float(series.median())
            }
        except Exception:
            continue
    return result

def _prepare_boxplot_data(df, numeric_columns):
    result = {}
    for column in numeric_columns:
        try:
            series = pd.to_numeric(df[column], errors="coerce")
            series = (series.replace([np.inf, -np.inf], np.nan).dropna())
            if len(series) == 0:
                continue
            if series.nunique() <= 1:
                continue
            q1 = float(series.quantile(0.25))
            median = float(series.quantile(0.50))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_fence = (q1 - 1.5 * iqr)
            upper_fence = (q3 + 1.5 * iqr)
            lower_values = series[series >= lower_fence]
            upper_values = series[series <= upper_fence]
            lower = (float(lower_values.min()) if len(lower_values) > 0 else float(series.min()))
            upper = (float(upper_values.max()) if len(upper_values) > 0 else float(series.max()))
            outliers = series[(series < lower_fence) | (series > upper_fence)]
            if len(outliers) > MAX_BOX_OUTLIERS:
                outliers = outliers.sample(n=MAX_BOX_OUTLIERS, random_state=42)
            result[str(column)] = {
                "min":
                    float(series.min()),
                "q1":
                    q1,
                "median":
                    median,
                "q3":
                    q3,
                "max":
                    float(series.max()),
                "lower":
                    lower,
                "upper":
                    upper,
                "mean":
                    float(series.mean()),
                "outliers": [float(value) for value in outliers]
            }
        except Exception:
            continue
    return result

def _prepare_categorical_chart_data(df, categorical_columns, max_categories=MAX_CATEGORIES):
    result = {}
    for column in categorical_columns:
        try:
            series = (df[column].dropna().astype(str))
            if len(series) == 0:
                continue
            counts = (series.value_counts().head(max_categories))
            if len(counts) == 0:
                continue
            result[str(column)] = {"categories": [str(value) for value in counts.index], "counts": [int(value) for value in counts.values]}
        except Exception:
            continue
    return result

def _prepare_scatter_chart_data(df, numeric_columns, max_variables=MAX_SCATTER_VARIABLES, max_points=MAX_SCATTER_POINTS):
    selected_columns = numeric_columns[:max_variables]
    result = []
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            column_x = selected_columns[i]
            column_y = selected_columns[j]
            try:
                pair = df[[column_x, column_y]].copy()
                pair = (pair.replace([np.inf, -np.inf], np.nan).dropna())
                if len(pair) < 2:
                    continue
                if (pair[column_x].nunique() <= 1 or pair[column_y].nunique() <= 1):
                    continue
                if len(pair) > max_points:
                    pair = pair.sample(n=max_points, random_state=42)
                points = []
                for _, row in pair.iterrows():
                    try:
                        points.append({"x": float(row[column_x]), "y": float(row[column_y])})
                    except (TypeError, ValueError):
                        continue
                if points:
                    result.append({"x_column": str(column_x), "y_column": str(column_y), "points": points})
            except Exception:
                continue
    return result

def _prepare_correlation_chart_data(correlation, method="spearman", max_columns=MAX_HEATMAP_COLUMNS):
    if not correlation:
        return None
    method_result = correlation.get(method)
    if not method_result:
        return None
    matrix = method_result.get("correlation")
    if matrix is None:
        return None
    if not isinstance(matrix, pd.DataFrame):
        return None
    if matrix.empty:
        return None
    columns = list(matrix.columns)
    if len(columns) > max_columns:
        columns = columns[:max_columns]
        matrix = matrix.loc[columns, columns]
    values = []
    for row_index in range(len(columns)):
        for column_index in range(len(columns)):
            value = matrix.iloc[row_index, column_index]
            if pd.isna(value):
                continue
            values.append({"x": int(column_index), "y": int(row_index), "value": float(value)})
    return {"method": method, "columns": [str(column) for column in columns], "values": values}

def _prepare_time_analysis_chart_data(time_analysis):
    result = {"hourly": {}, "weekday": {}, "daily": {}}
    if not time_analysis:
        return result
    analyses = time_analysis.get("analyses", {})
    if not isinstance(analyses, dict):
        return result
    for column, analysis in analyses.items():
        if not isinstance(analysis, dict):
            continue
        column_name = str(column)
        hourly = analysis.get("hourly_distribution", {})
        if (isinstance(hourly, dict) and hourly.get("status") == "ok"):
            records = hourly.get("results", [])
            if records:
                result["hourly"][column_name] = {"labels": [int(item["hour"]) for item in records], "counts": [int(item["count"]) for item in records]}
        weekday = analysis.get("weekday_distribution", {})
        if (isinstance(weekday, dict) and weekday.get("status") == "ok"):
            records = weekday.get("results", [])
            if records:
                result["weekday"][column_name] = {"labels": [str(item["weekday"]) for item in records], "counts": [int(item["count"]) for item in records]}
        daily = analysis.get("daily_distribution", {})
        if (isinstance(daily, dict) and daily.get("status") == "ok"):
            records = daily.get("results", [])
            if records:
                records = records[:MAX_TIME_POINTS]
                result["daily"][column_name] = {"labels": [str(item["date"]) for item in records], "counts": [int(item["count"]) for item in records]}
    return result

def _prepare_time_series_chart_data(time_series):
    result = {}
    if not time_series:
        return result
    if time_series.get("status") != "ok":
        return result
    aggregation = time_series.get("aggregation")
    if not aggregation:
        return result
    if aggregation.get("status") != "ok":
        return result
    frequency = aggregation.get("frequency", "unknown")
    variables = aggregation.get("variables", {})
    if not isinstance(variables, dict):
        return result
    for column, variable_result in variables.items():
        if not isinstance(variable_result, dict):
            continue
        if variable_result.get("status") != "ok":
            continue
        records = variable_result.get("results", [])
        if not records:
            continue
        records = records[:MAX_TIME_POINTS]
        data = []
        for item in records:
            time_key = "datetime" if "datetime" in item else "time_index"
            if time_key not in item:
                continue
            if frequency == "irregular":
                value = item.get("value")
            else:
                value = item.get("mean")
            if value is None:
                continue
            try:
                data.append({"datetime": str(item[time_key]), "value": float(value)})
            except (TypeError, ValueError):
                continue
        if data:
            result[str(column)] = {"frequency": str(frequency), "data": data}
    return result

def _prepare_dashboard_visualizations(df, variable_types, correlation=None, time_analysis=None, time_series=None):
    numeric_columns = _get_numeric_columns(df, variable_types)
    categorical_columns = _get_categorical_columns(df, variable_types)
    return {
        "histograms":
            _prepare_histogram_data(df, numeric_columns),
        "boxplots":
            _prepare_boxplot_data(df, numeric_columns),
        "categorical_bars":
            _prepare_categorical_chart_data(df, categorical_columns),
        "scatter_plots":
            _prepare_scatter_chart_data(df, numeric_columns),
        "correlation_heatmap":
            _prepare_correlation_chart_data(correlation),
        "time_analysis":
            _prepare_time_analysis_chart_data(time_analysis),
        "time_series":
            _prepare_time_series_chart_data(time_series)
    }

def _prepare_analysis_data(
    df,
    profile=None,
    statistics=None,
    quality=None,
    missingness=None,
    time_analysis=None,
    time_series=None,
    correlation=None,
    image_data=None
):
    variable_types = {}
    if profile:
        variable_types = profile.get("variable_types", {})
    return {
        "dataset":
            _prepare_dataset_summary(df),
        "variables":
            _prepare_variable_summary(df, variable_types),
        "numeric_data":
            _prepare_numeric_data(df, variable_types),
        "categorical_data":
            _prepare_categorical_data(df, variable_types),
        "identifier_data":
            _prepare_identifier_data(df, variable_types, profile),
        "identifier_combinations":
            _prepare_identifier_combinations(df, variable_types, profile),
        "identifier_chart_data":
            _prepare_identifier_chart_data(df, variable_types, profile),
        "time_data":
            _prepare_time_data(df, variable_types),
        "statistics":
            statistics or {},
        "quality":
            quality or {},
        "missingness":
            missingness or {},
        "time_analysis":
            time_analysis or {},
        "time_series":
            time_series or {},
        "correlation":
            correlation or {},
        "visualizations":
            _prepare_dashboard_visualizations(
                df=df,
                variable_types=variable_types,
                correlation=correlation,
                time_analysis=time_analysis,
                time_series=time_series
            ),
        "images":
            image_data or {}
    }

def _load_template(template_path):
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dashboard template not found: {path}"
        )
    return path.read_text(encoding="utf-8")

def generate_dashboard(
    df,
    profile=None,
    statistics=None,
    quality=None,
    missingness=None,
    time_analysis=None,
    time_series=None,
    correlation=None,
    visualizations=None,
    output_path="output/dashboard.html",
    template_path="report/dashboard_template.html",
    title="Interactive Data Dashboard"
):
    """生成独立的交互式 HTML Dashboard。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = _load_template(template_path)
    image_data = _build_image_data(visualizations)
    dashboard_data = _prepare_analysis_data(
        df=df,
        profile=profile,
        statistics=statistics,
        quality=quality,
        missingness=missingness,
        time_analysis=time_analysis,
        time_series=time_series,
        correlation=correlation,
        image_data=image_data
    )
    html_content = template.replace("{{TITLE}}", html.escape(str(title)))
    dashboard_json = _to_json(dashboard_data)
    html_content = html_content.replace("{{DASHBOARD_DATA_JSON}}", dashboard_json)
    html_content = html_content.replace("{{TITLE}}", html.escape(str(title)))
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
