from pathlib import Path
import base64
import json
import html
import math
from datetime import date, datetime
import numpy as np
import pandas as pd
from tools.data_profiler import parse_datetime_flexible
MAX_CATEGORIES = 15
MAX_HEATMAP_COLUMNS = 30
# 全量导出与抽样的分界行数：不超过该值时始终写入全部行；
# 超过时默认改为代表性抽样，样本规模也取该值（列式存储下 50 万行约 40 MB，可负担）。
FULL_EXPORT_ROW_LIMIT = 500000
MAX_IDENTIFIER_VALUES = 300000
MAX_IDENTIFIER_COMBINATION_VALUES = 300000
MAX_IDENTIFIER_COMBINATION_FIELDS = 4

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
    parsed = parse_datetime_flexible(series)
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return ((parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1))

def _representative_positions(count, limit, rng=None):
    """把有序序列 [0, count) 均分成 limit 段，每段内随机取一个位置。

    返回升序、去重后的位置数组。相比纯随机抽样，等距分层保证抽样点铺满
    整个序列（不会聚堆、也不会留下空档）；段内随机则避免被数据自身的
    周期性规律带偏。
    """
    if (limit is None or limit <= 0 or count <= limit):
        return np.arange(count)
    # 用 limit+1 个切点把 [0, count) 切成 limit 段；每段至少给出一个候选位置，
    # 各段区间互不重叠，因此结果一定是 limit 个互不相同的行。
    edges = (np.linspace(0, count, limit + 1))
    starts = (np.floor(edges[:-1]).astype(int))
    ends = (np.ceil(edges[1:]).astype(int))
    starts = (np.minimum(starts, count - 1))
    ends = (np.minimum(np.maximum(ends, starts + 1), count))
    positions = []
    for start, end in zip(starts, ends):
        if rng is None:
            positions.append((start + end - 1) // 2)
        else:
            positions.append(int(rng.integers(start, end)))
    return np.unique(np.asarray(positions, dtype=int))


def _sort_positions(values):
    """返回按 values 升序排列的行位置；无法解析（NaN）的行排最后并剔除。"""
    numeric = (np.asarray(values, dtype="float64"))
    valid = (np.isfinite(numeric))
    order = (np.argsort(numeric, kind="stable"))
    if valid.all():
        return order
    return order[valid[order]]


def _allocate_shares(sizes, limit):
    """用最大余额法把 limit 个名额按比例分配给各组。"""
    total = (float(sum(sizes)))
    if total <= 0:
        return ([0] * len(sizes))
    exact = ([limit * size / total for size in sizes])
    base = ([int(math.floor(value)) for value in exact])
    remaining = (limit - sum(base))
    if remaining <= 0:
        return base
    order = (sorted(range(len(sizes)), key=lambda index: exact[index] - base[index], reverse=True))
    position = 0
    guard = 0
    limit_guard = (len(order) * (limit + 1))
    while (remaining > 0 and guard < limit_guard):
        index = (order[position % len(order)])
        if base[index] < int(sizes[index]):
            base[index] += 1
            remaining -= 1
        position += 1
        guard += 1
    return base


def _to_order_keys(values):
    """把排序键统一转成 float64 数组；无法解析的位置记为 +inf（排到最后）。"""
    if values is None:
        return None
    try:
        series = (pd.Series(values).reset_index(drop=True))
    except Exception:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        deltas = (series - pd.Timestamp("1970-01-01"))
        series = (deltas.dt.total_seconds() * 1000.0)
    numeric = (pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64"))
    return np.where(np.isnan(numeric), np.inf, numeric)


def _representative_sample(data, limit, order_values=None, group_columns=None):
    """抽取一份尽量代表整体趋势与分布的子集。

    两层分层：
      1. 有分类变量时按类别组合分配名额（比例分层），
         样本里的类别构成与原始数据保持一致，小类别不会被抽没；
      2. 每个类别内部（或没有类别时的整体）按时间升序排列后等距分层、
         层内随机取一行，保证抽样点覆盖完整时间跨度，
         趋势不会被截断、也不会因随机波动而忽高忽低。

    没有时间变量时退化为纯随机抽样——此时没有趋势需要保留，
    随机抽样本身对分布就是无偏的。
    """
    count = (len(data))
    if (limit is None or count <= limit):
        return data
    if order_values is None:
        return data.sample(n=limit, random_state=42)

    keys = (_to_order_keys(order_values))
    rng = (np.random.default_rng(42))
    columns = [column for column in (group_columns or []) if column in data.columns]

    if (not columns):
        # 无分类变量：整体按时间等距分层抽样。
        order = (_sort_positions(keys))
        return data.iloc[order[_representative_positions(len(order), limit, rng)]]

    # 有分类变量：先按 (类别组合, 时间) 稳定排序，再在每组内等距分层抽样。
    labels = (data[columns[0]].astype(str).to_numpy())
    for column in columns[1:]:
        labels = (labels + "\u0001" + data[column].astype(str).to_numpy())
    order = (np.lexsort((keys, labels)))
    valid = (np.isfinite(keys))
    order = (order[valid[order]])
    if not len(order):
        return data

    sorted_labels = (labels[order])
    change = (np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1)
    starts = (np.concatenate(([0], change)))
    ends = (np.concatenate((change, [len(sorted_labels)])))
    shares = (_allocate_shares((ends - starts).tolist(), limit))

    picked = []
    for start, end, amount in zip(starts, ends, shares):
        if amount <= 0:
            continue
        block = (order[start:end])
        if amount >= len(block):
            picked.append(block)
        else:
            picked.append(block[_representative_positions(len(block), amount, rng)])
    if not picked:
        return data
    return data.iloc[np.sort(np.concatenate(picked))]


def _resolve_order_values(df, variable_types):
    """返回用于等距抽样的排序键（时间值）；没有时间变量时返回 None。"""
    time_type, time_column = (_resolve_time_column(df, variable_types))
    if (time_column is None or time_column not in df.columns):
        return None
    try:
        return _time_series_values(df[time_column], time_type)
    except Exception:
        return None


def _resolve_group_columns(df, variable_types, limit):
    """挑选用于比例分层的分类变量；没有合适的就返回空列表。

    从「取值个数最多」的列开始逐个加入，只要累计组合数不超过上限就继续。
    取值越多的分组越容易在随机抽样里漏掉小类别，最需要按比例保底；
    但组合数太多会把样本切得太碎，每组只剩一两行，反而破坏组内的时间覆盖。
    """
    if (not limit or limit <= 0):
        return []
    max_combinations = (max(2, int(limit // 5)))
    candidates = []
    for column in _get_categorical_columns(df, variable_types):
        try:
            unique_count = (int(df[column].nunique(dropna=True)))
        except Exception:
            continue
        if (unique_count > 1):
            candidates.append((column, unique_count))
    # 取值个数从多到少：优先保住最细的分组维度。
    candidates.sort(key=lambda item: item[1], reverse=True)
    chosen = []
    total = 1
    for column, unique_count in candidates:
        if (total * unique_count > max_combinations):
            continue
        chosen.append(column)
        total *= unique_count
    return chosen


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

def _prepare_table_data(df, variable_types, profile=None, max_points=FULL_EXPORT_ROW_LIMIT, full_data=False):
    """交互式报告唯一的「一张表」——行级明细只存这一份。

    采用列式存储 + 整数编码（分类列、标识符列的值字典各存一次），
    相比逐行的 JSON 对象数组可省约 4 倍体积。
    数值分布、变量关系、时间变化、归一化、按类别统计、标识符筛选
    等所有交互图表都从这一份数据派生，不再各存一份行数据。
    """
    numeric_columns = [
        column
        for column, info
        in (variable_types or {}).items()
        if (info.get("detected_type") == "numeric" and column in df.columns)
    ]
    categorical_columns = list(_get_categorical_columns(df, variable_types))
    identifier_columns = list(_get_identifier_columns(df, variable_types, profile))
    time_type, time_column = _resolve_time_column(df, variable_types)
    columns_to_use = list(dict.fromkeys(
        list(numeric_columns) + categorical_columns + identifier_columns
    ))
    if (time_column is not None and time_column in df.columns and time_column not in columns_to_use):
        columns_to_use.append(time_column)
    empty_payload = {
        "available": False,
        "row_count": 0,
        "source_row_count": int(len(df)),
        "truncated": False,
        "sample_limit": (None if full_data else max_points),
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "identifier_columns": [str(column) for column in identifier_columns],
        "time_type": time_type,
        "time_column": (str(time_column) if time_column is not None else None),
        "numeric": {},
        "categorical": {},
        "identifiers": {},
        "time_values": []
    }
    if (df.empty or not columns_to_use):
        return empty_payload
    data = df[columns_to_use].copy()
    source_row_count = int(len(data))
    truncated = False
    if (not full_data and max_points and len(data) > max_points):
        # 抽一份代表性子集：先按时间等距分层，再按分类组合比例保底。
        data = _representative_sample(
            data,
            max_points,
            order_values=_resolve_order_values(data, variable_types),
            group_columns=_resolve_group_columns(data, variable_types, max_points)
        )
        truncated = True
    data = data.replace([np.inf, -np.inf], np.nan)
    payload = dict(empty_payload)
    payload.update({
        "available": True,
        "row_count": int(len(data)),
        "source_row_count": source_row_count,
        "truncated": bool(truncated)
    })
    for column in numeric_columns:
        try:
            series = pd.to_numeric(data[column], errors="coerce")
            payload["numeric"][str(column)] = [None if pd.isna(value) else float(value) for value in series]
        except Exception:
            payload["numeric"][str(column)] = [None] * len(data)
    for key, columns in (("categorical", categorical_columns), ("identifiers", identifier_columns)):
        for column in columns:
            try:
                codes, uniques = pd.factorize(data[column].astype("string").fillna("缺失"), sort=False)
                payload[key][str(column)] = {
                    "values": [str(value) for value in uniques],
                    "codes": [int(code) for code in codes]
                }
            except Exception:
                continue
    if (time_column is not None and time_column in data.columns):
        try:
            values = _time_series_values(data[time_column], time_type)
            payload["time_values"] = [None if pd.isna(value) else float(value) for value in values]
        except Exception:
            payload["time_values"] = [None] * len(data)
    return payload

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

def _get_numeric_columns(df, variable_types):
    return [column for column, info in variable_types.items() if (info.get("detected_type") == "numeric" and column in df.columns)]

def _get_categorical_columns(df, variable_types):
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    return [column for column, info in variable_types.items() if (info.get("detected_type") in categorical_types and column in df.columns)]

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

def _prepare_analysis_data(
    df,
    profile=None,
    statistics=None,
    quality=None,
    missingness=None,
    correlation=None,
    image_data=None,
    export_all_data=False
):
    variable_types = {}
    if profile:
        variable_types = profile.get("variable_types", {})
    table_payload = _prepare_table_data(df, variable_types, profile, full_data=export_all_data)
    return {
        "dataset":
            _prepare_dataset_summary(df),
        "variables":
            _prepare_variable_summary(df, variable_types),
        "table_data":
            table_payload,
        "categorical_data":
            _prepare_categorical_data(df, variable_types),
        "identifier_data":
            _prepare_identifier_data(df, variable_types, profile),
        "identifier_combinations":
            _prepare_identifier_combinations(df, variable_types, profile),
        "statistics":
            statistics or {},
        "quality":
            quality or {},
        "missingness":
            missingness or {},
        "correlation":
            correlation or {},
        "data_scope":
            {
                "export_all_data": bool(export_all_data),
                "source_row_count": int(len(df)),
                "sample_limit": (None if export_all_data else FULL_EXPORT_ROW_LIMIT),
                "numeric_truncated": bool(table_payload.get("truncated")),
                "numeric_row_count": int(table_payload.get("row_count") or 0),
                "time_truncated": bool(table_payload.get("truncated")),
                "time_row_count": int(table_payload.get("row_count") or 0),
                "sampled": bool(not export_all_data and table_payload.get("truncated"))
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
    correlation=None,
    visualizations=None,
    output_path="output/dashboard.html",
    template_path="report/dashboard_template.html",
    title="Interactive Data Dashboard",
    export_all_data=True
):
    """生成独立的交互式 HTML Dashboard。

    export_all_data=True（默认）时，交互式图表使用完整数据，不再按 5000 行等上限抽样。
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
