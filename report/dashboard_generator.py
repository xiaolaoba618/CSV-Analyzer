from pathlib import Path
import base64
import json
import html
import math
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

def _resolve_time_column(df, variable_types):
    """返回 (time_type, time_column)，优先使用 datetime，其次 time_index。"""
    datetime_columns = [
        column
        for column, info
        in (variable_types or {}).items()
        if (info.get("detected_type") == "datetime" and column in df.columns)
    ]
    time_index_columns = [
        column
        for column, info
        in (variable_types or {}).items()
        if (info.get("detected_type") == "time_index" and column in df.columns)
    ]
    if datetime_columns:
        column = ("DATETIME" if "DATETIME" in datetime_columns else datetime_columns[0])
        return ("datetime", column)
    if time_index_columns:
        return ("time_index", time_index_columns[0])
    return (None, None)

def _time_series_values(series, time_type):
    """把时间列转换为可比较的数值：datetime 用毫秒时间戳，time_index 用浮点数值。"""
    if time_type == "time_index":
        return pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return ((parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1))

def _prepare_time_filter(df, variable_types):
    """准备时间范围筛选所需的元数据（不改变原有数据结构，仅新增）。"""
    time_type, time_column = _resolve_time_column(df, variable_types)
    if time_type is None or time_column is None:
        return {"available": False, "time_type": None, "time_column": None, "min": None, "max": None}
    try:
        values = _time_series_values(df[time_column], time_type).dropna()
    except Exception:
        values = None
    if values is None or len(values) == 0:
        return {"available": False, "time_type": str(time_type), "time_column": str(time_column), "min": None, "max": None}
    return {
        "available": True,
        "time_type": str(time_type),
        "time_column": str(time_column),
        "min": float(values.min()),
        "max": float(values.max())
    }

def _prepare_table_preview(df, max_rows=10):
    """准备数据表预览（前 max_rows 行），用于 Dashboard 中的表格预览面板。"""
    if df is None or df.empty:
        return {"columns": [], "rows": []}
    head = df.head(max_rows)
    columns = [str(column) for column in head.columns]
    rows = []
    for _, row in head.iterrows():
        rows.append([_make_json_safe(row[column]) for column in head.columns])
    return {"columns": columns, "rows": rows}

def _prepare_numeric_data(df, variable_types, max_points=MAX_NUMERIC_POINTS, full_data=False):
    numeric_columns = [
        column
        for column, info
        in variable_types.items()
        if (info.get("detected_type") == "numeric" and column in df.columns)
    ]
    if not numeric_columns:
        return {
            "columns": [],
            "rows": [],
            "source_row_count": int(len(df)),
            "row_count": 0,
            "truncated": False,
            "sample_limit": (None if full_data else MAX_NUMERIC_POINTS)
        }
    categorical_columns = _get_categorical_columns(df, variable_types)
    time_type, time_column = _resolve_time_column(df, variable_types)
    # 保持采样行的分类字段与数值字段对应。
    columns_to_use = list(dict.fromkeys(numeric_columns + categorical_columns))
    # 时间列仅用于生成逐行可比较的时间值，不作为数值/分类变量暴露。
    if time_column is not None and time_column not in columns_to_use:
        columns_to_use = columns_to_use + [time_column]
    data = df[columns_to_use].copy()
    source_row_count = int(len(data))
    if full_data or max_points is None:
        truncated = False
    elif len(data) > max_points:
        # 默认路径：超过上限时随机抽样子集，供交互图表使用。
        data = data.sample(n=max_points, random_state=42)
        truncated = True
    else:
        truncated = False
    data = data.replace([np.inf, -np.inf], np.nan)
    time_values = None
    if time_column is not None:
        try:
            time_values = _time_series_values(data[time_column], time_type)
        except Exception:
            time_values = None
    rows = []
    for position, (_, row) in enumerate(data.iterrows()):
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
        if time_values is not None:
            try:
                time_value = time_values.iloc[position]
            except (KeyError, IndexError):
                time_value = None
            item["__time__"] = (None if time_value is None or pd.isna(time_value) else float(time_value))
        rows.append(item)
    return {
        "columns": [str(column) for column in numeric_columns],
        "rows": rows,
        "source_row_count": source_row_count,
        "row_count": int(len(rows)),
        "truncated": bool(truncated),
        "sample_limit": (None if full_data else MAX_NUMERIC_POINTS)
    }

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

def _prepare_time_data(df, variable_types, max_points=MAX_TIME_POINTS, full_data=False):
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
        source_row_count = int(len(data))
        if full_data or max_points is None:
            truncated = False
        elif len(data) > max_points:
            index = np.linspace(0, len(data) - 1, max_points, dtype=int)
            data = data.iloc[index]
            truncated = True
        else:
            truncated = False
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
            "rows": rows,
            "source_row_count": source_row_count,
            "row_count": int(len(rows)),
            "truncated": bool(truncated),
            "sample_limit": (None if full_data else MAX_TIME_POINTS)
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
    source_row_count = int(len(data))
    if full_data or max_points is None:
        truncated = False
    elif len(data) > max_points:
        index = np.linspace(0, len(data) - 1, max_points, dtype=int)
        data = data.iloc[index]
        truncated = True
    else:
        truncated = False
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
        "rows": rows,
        "source_row_count": source_row_count,
        "row_count": int(len(rows)),
        "truncated": bool(truncated),
        "sample_limit": (None if full_data else MAX_TIME_POINTS)
    }

def _get_numeric_columns(df, variable_types):
    return [column for column, info in variable_types.items() if (info.get("detected_type") == "numeric" and column in df.columns)]

def _get_categorical_columns(df, variable_types):
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    return [column for column, info in variable_types.items() if (info.get("detected_type") in categorical_types and column in df.columns)]

def _prepare_histogram_data(df, numeric_columns, bins=30, max_points=MAX_HISTOGRAM_POINTS, full_data=False):
    result = {}
    for column in numeric_columns:
        try:
            series = pd.to_numeric(df[column], errors="coerce")
            series = (series.replace([np.inf, -np.inf], np.nan).dropna())
            if len(series) == 0:
                continue
            if not full_data and max_points is not None and len(series) > max_points:
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

def _prepare_boxplot_data(df, numeric_columns, full_data=False):
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
            if not full_data and len(outliers) > MAX_BOX_OUTLIERS:
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

def _prepare_dashboard_visualizations(df, variable_types, correlation=None, time_analysis=None, time_series=None, full_data=False):
    numeric_columns = _get_numeric_columns(df, variable_types)
    categorical_columns = _get_categorical_columns(df, variable_types)
    return {
        "histograms":
            _prepare_histogram_data(df, numeric_columns, full_data=full_data),
        "boxplots":
            _prepare_boxplot_data(df, numeric_columns, full_data=full_data),
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
    image_data=None,
    export_all_data=False
):
    variable_types = {}
    if profile:
        variable_types = profile.get("variable_types", {})
    numeric_payload = _prepare_numeric_data(df, variable_types, full_data=export_all_data)
    time_payload = _prepare_time_data(df, variable_types, full_data=export_all_data)
    return {
        "dataset":
            _prepare_dataset_summary(df),
        "variables":
            _prepare_variable_summary(df, variable_types),
        "numeric_data":
            numeric_payload,
        "categorical_data":
            _prepare_categorical_data(df, variable_types),
        "identifier_data":
            _prepare_identifier_data(df, variable_types, profile),
        "identifier_combinations":
            _prepare_identifier_combinations(df, variable_types, profile),
        "identifier_chart_data":
            _prepare_identifier_chart_data(df, variable_types, profile),
        "time_data":
            time_payload,
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
                time_series=time_series,
                full_data=export_all_data
            ),
        "data_scope":
            {
                "export_all_data": bool(export_all_data),
                "source_row_count": int(len(df)),
                "sample_limit": (None if export_all_data else MAX_NUMERIC_POINTS),
                "numeric_truncated": bool(numeric_payload.get("truncated")),
                "numeric_row_count": int(numeric_payload.get("row_count") or 0),
                "time_truncated": bool(time_payload.get("truncated")),
                "time_row_count": int(time_payload.get("row_count") or 0),
                "sampled": bool(
                    not export_all_data
                    and (numeric_payload.get("truncated") or time_payload.get("truncated"))
                )
            },
        "time_filter":
            _prepare_time_filter(df, variable_types),
        "table_preview":
            _prepare_table_preview(df),
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
    title="Interactive Data Dashboard",
    export_all_data=False
):
    """生成独立的交互式 HTML Dashboard。

    export_all_data=True 时，交互式图表使用完整数据，不再按 5000 行等上限抽样。
    """
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
        image_data=image_data,
        export_all_data=export_all_data
    )
    html_content = template.replace("{{TITLE}}", html.escape(str(title)))
    dashboard_json = _to_json(dashboard_data)
    html_content = html_content.replace("{{DASHBOARD_DATA_JSON}}", dashboard_json)
    html_content = html_content.replace("{{TITLE}}", html.escape(str(title)))
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


ESTIMATE_SAMPLE_ROWS = 400
# 与数据量基本无关的固定开销（模板 + 统计量 + 变量卡片等），按实测校准。
ESTIMATE_BASE_BYTES = 500_000
# 静态图片每张的估算开销（base64 PNG），按实测校准。
# PNG 体积随数据离散程度浮动很大（同点数的散点图可差数倍），因此只给区间。
ESTIMATE_IMAGE_HISTOGRAM_BYTES = 60_000
ESTIMATE_IMAGE_BOXPLOT_BYTES = 50_000
ESTIMATE_IMAGE_SCATTER_BYTES = 400_000
ESTIMATE_IMAGE_CATEGORY_BYTES = 100_000
# 时间序列趋势图依赖全量聚合结果，无法用预览实测，按实测校准。
ESTIMATE_IMAGE_TIMESERIES_BYTES = 180_000
# 小时/星期/每日分布三张图的固定开销。
ESTIMATE_IMAGE_FIXED_BYTES = 240_000
# 预览只见局部主体，全量数据跨主体的跳变会额外产生异常点，按实测放大。
ESTIMATE_ANOMALY_RATE_SAFETY = 1.6
ESTIMATE_ANOMALY_RATE_CAP = 0.25
# 预估上下界系数：静态图片占比越高、数据越杂乱，不确定性越大。
ESTIMATE_RANGE_LOW_FACTOR = 0.75
ESTIMATE_RANGE_HIGH_FACTOR = 1.6
# 以下常量仅在无法从预览样本实测时兜底使用。
ESTIMATE_TIMESERIES_BYTES_PER_RECORD_FALLBACK = 100
ESTIMATE_TIME_CHART_BYTES_PER_RECORD_FALLBACK = 48
ESTIMATE_ANOMALY_BYTES_PER_ITEM_FALLBACK = 60
ESTIMATE_ANOMALY_ROW_RATE_FALLBACK = 0.05
ESTIMATE_SCATTER_BYTES_PER_POINT_FALLBACK = 25
ESTIMATE_IDENTIFIER_BYTES_PER_ROW_FALLBACK = 80


def _json_size(payload):
    """按 Dashboard 实际写入方式序列化后的字节数。"""
    try:
        return len(_to_json(payload).encode("utf-8"))
    except Exception:
        return 0


def _measure_scatter_point_bytes(sample, numeric_columns):
    """实测散点图单个点的字节成本（散点图恒为每对变量 5000 点）。"""
    selected = numeric_columns[:MAX_SCATTER_VARIABLES]
    if len(selected) < 2:
        return None
    try:
        pair = sample[[selected[0], selected[1]]].copy()
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return None
    if len(pair) < 2:
        return None
    points = []
    for _, row in pair.head(200).iterrows():
        try:
            points.append({"x": float(row[selected[0]]), "y": float(row[selected[1]])})
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    return _json_size(points) / len(points)


def _measure_identifier_row_bytes(sample, variable_types):
    """实测标识符联动数据每行的字节成本。"""
    if not _get_identifier_columns(sample, variable_types):
        return 0
    try:
        payload = _prepare_identifier_chart_data(sample, variable_types)
    except Exception:
        return None
    rows = int(payload.get("row_count") or 0)
    if rows <= 0:
        return None
    total = _json_size(payload)
    # 预览样本的标识符取值远少于全量，扣除取值表本身的固定开销。
    values_size = _json_size(payload.get("identifier_columns") or {})
    return max(total - values_size, 0) / rows


def _measure_time_series_costs(sample, variable_types, numeric_columns):
    """实测时间序列相关记录的字节成本。

    返回 (每条聚合记录字节, 每条异常记录字节, 每行每列的异常率, 每条趋势图记录字节)。
    """
    fallback = (
        ESTIMATE_TIMESERIES_BYTES_PER_RECORD_FALLBACK,
        ESTIMATE_ANOMALY_BYTES_PER_ITEM_FALLBACK,
        ESTIMATE_ANOMALY_ROW_RATE_FALLBACK,
        ESTIMATE_TIME_CHART_BYTES_PER_RECORD_FALLBACK,
    )
    if not numeric_columns:
        return fallback
    try:
        from tools.time_series import analyze_time_series
        result = analyze_time_series(sample, variable_types)
    except Exception:
        return fallback
    if not isinstance(result, dict) or result.get("status") != "ok":
        return fallback
    sample_rows = int(len(sample))
    aggregation = result.get("aggregation") or {}
    variables = aggregation.get("variables") or {}
    records = 0
    records_bytes = 0
    for variable_result in variables.values():
        if not isinstance(variable_result, dict):
            continue
        items = variable_result.get("results") or []
        if not items:
            continue
        records += len(items)
        records_bytes += _json_size(items)
    anomaly_count = 0
    anomaly_bytes = 0
    original = result.get("original_series") or {}
    for variable_result in (original.get("variables") or {}).values():
        if not isinstance(variable_result, dict):
            continue
        items = ((variable_result.get("anomalies") or {}).get("anomalies") or [])
        if not items:
            continue
        anomaly_count += len(items)
        anomaly_bytes += _json_size(items)
    bytes_per_record = (
        records_bytes / records
        if records
        else ESTIMATE_TIMESERIES_BYTES_PER_RECORD_FALLBACK
    )
    bytes_per_anomaly = (
        anomaly_bytes / anomaly_count
        if anomaly_count
        else ESTIMATE_ANOMALY_BYTES_PER_ITEM_FALLBACK
    )
    if sample_rows > 0 and len(numeric_columns) > 0:
        row_anomaly_rate = anomaly_count / float(sample_rows * len(numeric_columns))
    else:
        row_anomaly_rate = ESTIMATE_ANOMALY_ROW_RATE_FALLBACK
    # 趋势图数据（_prepare_time_series_chart_data 的输出）。
    chart_records = 0
    chart_bytes = 0
    try:
        chart = _prepare_time_series_chart_data(result)
        for payload in chart.values():
            items = payload.get("data") or []
            if not items:
                continue
            chart_records += len(items)
            chart_bytes += _json_size(items)
    except Exception:
        chart_records = 0
    bytes_per_chart_record = (
        chart_bytes / chart_records
        if chart_records
        else ESTIMATE_TIME_CHART_BYTES_PER_RECORD_FALLBACK
    )
    return (bytes_per_record, bytes_per_anomaly, row_anomaly_rate, bytes_per_chart_record)


def measure_payload_costs(df, variable_types=None, sample_rows=ESTIMATE_SAMPLE_ROWS):
    """从预览样本实测单位字节成本，供体积预估复用（一次测量，多次估算）。

    返回的键：
      detail_bytes_per_row      行级明细（数值明细 + 时间明细）每行字节
      scatter_bytes_per_point   交互散点图每个点字节
      identifier_bytes_per_row  标识符联动每行字节
      timeseries_bytes_per_record / anomaly_bytes_per_item / anomaly_row_rate /
      time_chart_bytes_per_record
      numeric_count / categorical_count
    """
    costs = {}
    if df is None or getattr(df, "empty", True):
        return costs
    variable_types = variable_types or {}
    sample = df.head(min(sample_rows, int(len(df))))
    numeric_columns = _get_numeric_columns(sample, variable_types)
    categorical_columns = _get_categorical_columns(sample, variable_types)
    costs["numeric_count"] = len(numeric_columns)
    costs["categorical_count"] = len(categorical_columns)
    detail_per_row = 0.0
    try:
        numeric_rows = _prepare_numeric_data(sample, variable_types, full_data=True).get("rows") or []
        if numeric_rows:
            detail_per_row += _json_size(numeric_rows) / len(numeric_rows)
    except Exception:
        pass
    try:
        time_rows = _prepare_time_data(sample, variable_types, full_data=True).get("rows") or []
        if time_rows:
            detail_per_row += _json_size(time_rows) / len(time_rows)
    except Exception:
        pass
    costs["detail_bytes_per_row"] = detail_per_row
    scatter_per_point = _measure_scatter_point_bytes(sample, numeric_columns)
    costs["scatter_bytes_per_point"] = (
        scatter_per_point
        if scatter_per_point
        else ESTIMATE_SCATTER_BYTES_PER_POINT_FALLBACK
    )
    identifier_per_row = _measure_identifier_row_bytes(sample, variable_types)
    costs["identifier_bytes_per_row"] = (
        identifier_per_row
        if identifier_per_row
        else ESTIMATE_IDENTIFIER_BYTES_PER_ROW_FALLBACK
    )
    (
        costs["timeseries_bytes_per_record"],
        costs["anomaly_bytes_per_item"],
        costs["anomaly_row_rate"],
        costs["time_chart_bytes_per_record"],
    ) = _measure_time_series_costs(sample, variable_types, numeric_columns)
    return costs


def _estimate_time_buckets(df, variable_types, total_rows):
    """估算时间序列聚合后的「时间桶」数量。

    注意：时间桶数不等于总行数。面板数据（多主体 × 时间，如 500 只股票 × 698 个交易日）
    的行数远多于时间桶数；只有「每行一个时间点」时才等于总行数。
    """
    resolved = _resolve_time_column(df, variable_types)
    if resolved is None:
        return 0
    time_column = resolved[1]
    try:
        work = df.head(min(2000, int(len(df))))
        unique_count = int(work[time_column].nunique(dropna=True))
    except Exception:
        return 0
    sample_rows = int(len(work))
    if sample_rows <= 0:
        return 0
    if total_rows <= sample_rows:
        return int(min(unique_count, total_rows))
    if unique_count >= sample_rows:
        # 预览内时间戳互不重复：每行一个时间点，可直接按总行数外推。
        return int(total_rows)
    # 存在重复时间戳（面板数据）：预览可能只覆盖部分主体，
    # 取「预览去重数」与「预览行数」的较大值作为保守估计。
    return int(min(total_rows, max(unique_count, sample_rows)))


# 区间格式化时的取整粒度：单位 -> (除数, 单位名, 粒度, 小数位)。
_SIZE_RANGE_UNITS = (
    (1024 ** 3, "GB", 0.5, 1),
    (1024 ** 2, "MB", 5.0, 0),
    (1024.0, "KB", 5.0, 0),
    (1.0, "B", 10.0, 0),
)


def format_size_range(low_bytes, high_bytes):
    """把体积区间格式化为「35~80 MB」这样的文本；上下界接近时退化为单个值。

    上下界一律「向外」取整（下限向下、上限向上），保证取整不会收窄已有区间。
    """
    low = max(float(low_bytes or 0), 0.0)
    high = max(float(high_bytes or 0), low)
    divisor, unit, step, decimals = next(
        item for item in _SIZE_RANGE_UNITS if high >= item[0]
    )
    low_value = math.floor(low / divisor / step) * step
    high_value = math.ceil(high / divisor / step) * step
    if low_value == high_value:
        return f"{low_value:.{decimals}f} {unit}"
    return f"{low_value:.{decimals}f}~{high_value:.{decimals}f} {unit}"


def _estimate_dashboard_bytes(df, variable_types, export_all_data, total_rows, costs):
    """按报告的构成逐项累加，得到体积的量级估计（字节）。

    构成（与 nullable 的行数关系各不相同）：
      1. 固定开销 + 静态图片：随列数、图数增长；
      2. 行级明细（数值 + 时间）：受「导出全部数据」控制；
      3. 交互散点图：变量两两组合 × 5000 点，永不受选项影响；
      4. 交互趋势图数据：时间桶数（上限 5000）× 数值列数；
      5. 时间序列聚合 + 异常点：按全量行数计（跨主体跳变会产生额外异常点，
         预览样本估出的异常率会按实测系数放大）；
      6. 标识符联动明细：按全量行数嵌入（上限 200 万行），不受选项影响。

    静态 PNG 的压缩率随数据离散程度浮动很大，因此这里只求量级，
    最终在界面上以「区间」形式呈现，不承诺精确值。
    """
    numeric_count = int(costs.get("numeric_count") or 0)
    categorical_count = int(costs.get("categorical_count") or 0)
    detail_per_row = float(costs.get("detail_bytes_per_row") or 0.0)
    scatter_per_point = float(
        costs.get("scatter_bytes_per_point") or ESTIMATE_SCATTER_BYTES_PER_POINT_FALLBACK
    )
    identifier_per_row = float(
        costs.get("identifier_bytes_per_row") or ESTIMATE_IDENTIFIER_BYTES_PER_ROW_FALLBACK
    )
    ts_per_record = float(
        costs.get("timeseries_bytes_per_record") or ESTIMATE_TIMESERIES_BYTES_PER_RECORD_FALLBACK
    )
    anomaly_per_item = float(
        costs.get("anomaly_bytes_per_item") or ESTIMATE_ANOMALY_BYTES_PER_ITEM_FALLBACK
    )
    anomaly_row_rate = float(costs.get("anomaly_row_rate") or ESTIMATE_ANOMALY_ROW_RATE_FALLBACK)
    chart_per_record = float(
        costs.get("time_chart_bytes_per_record") or ESTIMATE_TIME_CHART_BYTES_PER_RECORD_FALLBACK
    )
    buckets = _estimate_time_buckets(df, variable_types, total_rows)
    has_identifier = bool(
        _get_identifier_columns(
            df.head(min(ESTIMATE_SAMPLE_ROWS, int(len(df)))), variable_types
        )
    )
    used_rows = total_rows if export_all_data else min(total_rows, MAX_NUMERIC_POINTS)
    scatter_variables = min(numeric_count, MAX_SCATTER_VARIABLES)
    scatter_pairs = max(scatter_variables * (scatter_variables - 1) // 2, 0)
    scatter_points = min(MAX_SCATTER_POINTS, max(total_rows, 0)) * scatter_pairs
    time_series_figures = min(numeric_count, MAX_SCATTER_VARIABLES)
    chart_records = min(buckets, MAX_TIME_POINTS) * numeric_count
    anomaly_rate = min(anomaly_row_rate * ESTIMATE_ANOMALY_RATE_SAFETY, ESTIMATE_ANOMALY_RATE_CAP)
    identifier_rows = min(total_rows, MAX_IDENTIFIER_CHART_ROWS) if has_identifier else 0
    image_bytes = (
        (ESTIMATE_IMAGE_HISTOGRAM_BYTES + ESTIMATE_IMAGE_BOXPLOT_BYTES) * numeric_count
        + ESTIMATE_IMAGE_CATEGORY_BYTES * categorical_count
        + ESTIMATE_IMAGE_SCATTER_BYTES * scatter_pairs
        + ESTIMATE_IMAGE_TIMESERIES_BYTES * time_series_figures
        + ESTIMATE_IMAGE_FIXED_BYTES
    )
    return (
        ESTIMATE_BASE_BYTES
        + image_bytes
        + scatter_per_point * scatter_points
        + chart_per_record * chart_records
        + ts_per_record * buckets * numeric_count
        + anomaly_per_item * anomaly_rate * total_rows * numeric_count
        + identifier_per_row * identifier_rows
        + detail_per_row * used_rows
    )


def estimate_dashboard_size_range(
    df, variable_types=None, export_all_data=False, total_rows=None, costs=None
):
    """估算生成报告的体积累级区间，返回 (下限字节, 上限字节)。

    只保证真实体积大概率落在区间内，不做精确预测：静态图片的 base64 体积
    会随数据形态上下浮动数倍，这是无法在选文件阶段算准的部分。
    失败或数据为空时返回 (0, 0)。
    """
    if df is None or getattr(df, "empty", True):
        return (0, 0)
    variable_types = variable_types or {}
    if total_rows is None:
        total_rows = int(len(df))
    total_rows = int(total_rows)
    if total_rows <= 0:
        return (0, 0)
    if costs is None:
        costs = measure_payload_costs(df, variable_types)
    try:
        size = _estimate_dashboard_bytes(
            df, variable_types, export_all_data, total_rows, costs
        )
    except Exception:
        return (0, 0)
    if size <= 0:
        return (0, 0)
    return (
        int(size * ESTIMATE_RANGE_LOW_FACTOR),
        int(size * ESTIMATE_RANGE_HIGH_FACTOR),
    )


def estimate_dashboard_size(
    df, variable_types=None, export_all_data=False, total_rows=None, costs=None
):
    """估算体积的单点值（区间中点），用于需要单个数字的场景。"""
    low, high = estimate_dashboard_size_range(
        df, variable_types, export_all_data, total_rows, costs
    )
    return int((low + high) / 2)


def format_size(bytes_value):
    """把字节数格式化为便于阅读的文本。"""
    size = float(bytes_value or 0)
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size:.0f} B"
