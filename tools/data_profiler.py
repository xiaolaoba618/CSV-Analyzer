import numpy as np
import pandas as pd
DATETIME_SAMPLE_SIZE = 100
NUMERIC_SAMPLE_SIZE = 200
DATETIME_SUCCESS_THRESHOLD = 0.80
TIME_SUCCESS_THRESHOLD = 0.80
NUMERIC_SUCCESS_THRESHOLD = 0.80
IDENTIFIER_UNIQUENESS_THRESHOLD = 0.95
IDENTIFIER_EXACT_NAME_KEYWORDS = {"id", "uid", "uuid", "identifier", "标识符", "编号", "用户id", "玩家id"}
IDENTIFIER_SUFFIXES = (" id", " uid", " uuid", " identifier", " 编号",)
TEXT_AVERAGE_LENGTH_THRESHOLD = 50
TEXT_UNIQUE_RATIO_THRESHOLD = 0.50
DATETIME_NAME_KEYWORDS = {"date", "日期", "datetime", "timestamp", "时间戳"}
TIME_INDEX_NAME_KEYWORDS = {
    "week", "weeks", "wk", "week number", "week_number",
    "周", "周数", "星期", "period", "period number", "period_number"
}
TIME_NAME_KEYWORDS = {"time", "时间", "timestamp", "datetime", "时刻"}
DURATION_NAME_KEYWORDS = {
    "duration",
    "minute",
    "minutes",
    "mintue",
    "mintues",
    "hour",
    "hours",
    "second",
    "seconds",
    "day",
    "days",
    "playtime",
    "play time",
    "play_time",
    "total time",
    "total_time",
    "session time",
    "session_time",
    "session duration",
    "session_duration",
    "elapsed time",
    "elapsed_time"
}
NUMERIC_NAME_KEYWORDS = {
    "level",
    "flow",
    "ntu",
    "ph",
    "clr",
    "cl2",
    "alum",
    "pressure",
    "temperature",
    "temp",
    "speed",
    "amount",
    "value",
    "rate",
    "concentration",
    "height",
    "weight",
    "distance",
    "volume",
    "count"
}
BINARY_NAME_KEYWORDS = {"pump duty", "switch", "status", "flag", "binary", "on/off", "yes/no"}
TEXT_NAME_KEYWORDS = {"remark", "remarks", "comment", "comments", "description", "note", "notes"}

def _normalize_column_name(column):
    return (str(column).strip().lower().replace("_", " ").replace("-", " "))

def _looks_like_identifier_name(column):
    raw = str(column).strip().lower()
    normalized = _normalize_column_name(column)
    compact = normalized.replace(" ", "")
    if normalized in IDENTIFIER_EXACT_NAME_KEYWORDS:
        return True
    if compact in {"id", "uid", "uuid", "identifier", "playeruid", "playerid",
                   "userid", "userid", "deviceid", "customerid", "accountid",
                   "sessionid", "orderid", "itemid", "recordid"}:
        return True
    if normalized.endswith(IDENTIFIER_SUFFIXES):
        return True
    if raw.endswith("_id") or raw.endswith("_uid") or raw.endswith("_uuid"):
        return True
    if raw.lower().endswith("id") and len(raw) > 2:
        return True
    if raw.lower().endswith("uid") and len(raw) > 3:
        return True
    return False

def _name_contains_keyword(column, keywords):
    name = _normalize_column_name(column)
    for keyword in keywords:
        if keyword in name:
            return True
    return False

def _detect_name_hint(column):
    if _name_contains_keyword(column, DATETIME_NAME_KEYWORDS):
        return "datetime"
    if _name_contains_keyword(column, TIME_INDEX_NAME_KEYWORDS):
        return "time_index"
    if _name_contains_keyword(column, DURATION_NAME_KEYWORDS):
        return "duration"
    if _name_contains_keyword(column, TIME_NAME_KEYWORDS):
        return "time"
    if _name_contains_keyword(column, TEXT_NAME_KEYWORDS):
        return "text"
    if _name_contains_keyword(column, BINARY_NAME_KEYWORDS):
        return "binary"
    if _name_contains_keyword(column, NUMERIC_NAME_KEYWORDS):
        return "numeric"
    return None

def _try_parse_datetime(series, allow_numeric_compact=False):
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        if not allow_numeric_compact:
            return False
        numeric_values = (pd.to_numeric(series.dropna(), errors="coerce").dropna())
        if len(numeric_values) == 0:
            return False
        integer_like = np.all(np.isclose(numeric_values, np.round(numeric_values)))
        if not integer_like:
            return False
        string_values = (numeric_values.astype(np.int64).astype(str))
        compact_date_ratio = (
            string_values
            .str.fullmatch(r"\d{8}")
            .mean()
        )
        if compact_date_ratio < DATETIME_SUCCESS_THRESHOLD:
            return False
        try:
            parsed = pd.to_datetime(string_values, errors="coerce", format="%Y%m%d")
            success_ratio = parsed.notna().mean()
            return (success_ratio >= DATETIME_SUCCESS_THRESHOLD)
        except Exception:
            return False
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = (non_null.astype(str).str.strip().head(DATETIME_SAMPLE_SIZE))
    if len(sample) == 0:
        return False
    compact_date_ratio = (
        sample.str.fullmatch(
            r"\d{8}"
        ).mean()
    )
    if compact_date_ratio >= DATETIME_SUCCESS_THRESHOLD:
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="%Y%m%d")
            success_ratio = parsed.notna().mean()
            if success_ratio >= DATETIME_SUCCESS_THRESHOLD:
                return True
        except Exception:
            pass
    datetime_pattern_ratio = (sample.str.contains(r"[-/.:]", regex=True).mean())
    if datetime_pattern_ratio < DATETIME_SUCCESS_THRESHOLD:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        success_ratio = parsed.notna().mean()
        return (success_ratio >= DATETIME_SUCCESS_THRESHOLD)
    except Exception:
        return False

def _try_parse_time(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = (non_null.astype(str).str.strip().head(DATETIME_SAMPLE_SIZE))
    if len(sample) == 0:
        return False
    success_count = 0
    for value in sample:
        try:
            pd.to_datetime(value, format="%H:%M:%S")
            success_count += 1
            continue
        except Exception:
            pass
        try:
            pd.to_datetime(value, format="%H:%M")
            success_count += 1
            continue
        except Exception:
            pass
    success_ratio = (success_count / len(sample))
    return (success_ratio >= TIME_SUCCESS_THRESHOLD)

def _try_parse_numeric(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = (non_null.astype(str).str.strip().head(NUMERIC_SAMPLE_SIZE))
    if len(sample) == 0:
        return False
    missing_tokens = {"", "-", "--", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "."}
    cleaned = sample.mask(sample.isin(missing_tokens), np.nan)
    numeric_values = pd.to_numeric(cleaned, errors="coerce")
    valid_count = int(numeric_values.notna().sum())
    success_ratio = (valid_count / len(cleaned) if len(cleaned) > 0 else 0)
    return (success_ratio >= NUMERIC_SUCCESS_THRESHOLD)

def _looks_like_identifier(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    unique_count = non_null.nunique()
    unique_ratio = (unique_count / len(non_null))
    return (unique_ratio >= IDENTIFIER_UNIQUENESS_THRESHOLD)

def _looks_like_text(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    text_lengths = (non_null.astype(str).str.len())
    average_length = text_lengths.mean()
    unique_ratio = (non_null.nunique() / len(non_null))
    if (average_length >= TEXT_AVERAGE_LENGTH_THRESHOLD):
        return True
    if (average_length >= 20 and unique_ratio >= TEXT_UNIQUE_RATIO_THRESHOLD):
        return True
    return False

def _detect_variable_type(series, column=None):
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"type": "empty", "reason": "no_valid_values"}
    name_hint = None
    if column is not None:
        name_hint = _detect_name_hint(column)
    # 标识符语义优先于数值/分类推断。
    if column is not None and _looks_like_identifier_name(column):
        return {"type": "identifier", "reason": "identifier_column_name"}
    if pd.api.types.is_bool_dtype(series):
        return {"type": "boolean", "reason": "boolean_dtype"}
    # 显式 DATE 列
    if name_hint == "datetime":
        if _try_parse_datetime(series, allow_numeric_compact=True):
            return {"type": "datetime", "reason": "datetime_column_name_and_values"}
    # 显式时间索引列
    if name_hint == "time_index":
        if _try_parse_numeric(series):
            return {"type": "time_index", "reason": "time_index_column_name_and_numeric_values"}
    # 显式持续时间列
    if name_hint == "duration":
        if _try_parse_numeric(series):
            return {"type": "numeric", "reason": "duration_column_name"}
    # 显式 TIME 列
    if name_hint == "time":
        if _try_parse_time(series):
            return {"type": "time", "reason": "time_column_name_and_values"}
        # 仅在列名具有明确时间语义时识别数字 HHMM。
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(values) > 0:
                integer_like = np.all(np.isclose(values, np.round(values)))
                within_time_range = (values >= 0).all() and (values <= 2359).all()
                if (integer_like and within_time_range):
                    return {"type": "time", "reason": "numeric_time_column"}
    # 常规日期时间识别
    if _try_parse_datetime(series):
        return {"type": "datetime", "reason": "datetime_pattern_detected"}
    # 常规时间识别
    if _try_parse_time(series):
        return {"type": "time", "reason": "time_pattern_detected"}
    if pd.api.types.is_numeric_dtype(series):
        unique_count = non_null.nunique()
        if name_hint == "numeric":
            return {"type": "numeric", "reason": "numeric_dtype_and_measurement_name"}
        if name_hint == "binary":
            return {"type": "binary", "reason": "binary_column_name"}
        if unique_count == 2:
            return {"type": "binary", "reason": "numeric_binary_variable"}
        if unique_count <= 10:
            return {"type": "categorical_numeric", "reason": "low_cardinality_numeric"}
        return {"type": "numeric", "reason": "numeric_dtype"}
    if _try_parse_numeric(series):
        if name_hint == "numeric":
            return {"type": "numeric", "reason": "numeric_values_and_measurement_name"}
        if name_hint == "binary":
            return {"type": "binary", "reason": "binary_column_name"}
        converted = pd.to_numeric(
            non_null
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": np.nan,
                    "-": np.nan,
                    "--": np.nan,
                    "NA": np.nan,
                    "N/A": np.nan,
                    "na": np.nan,
                    "n/a": np.nan,
                    "null": np.nan,
                    "NULL": np.nan,
                    "None": np.nan,
                    "none": np.nan,
                    ".": np.nan
                }
            ),
            errors="coerce"
        )
        unique_count = (converted.dropna().nunique())
        if unique_count == 2:
            return {"type": "binary", "reason": "numeric_binary_stored_as_text"}
        if unique_count <= 10:
            return {"type": "categorical_numeric", "reason": "low_cardinality_numeric_stored_as_text"}
        return {"type": "numeric", "reason": "numeric_values_stored_as_text"}
    if name_hint == "text":
        return {"type": "text", "reason": "text_column_name"}
    if _looks_like_identifier(series):
        return {"type": "identifier", "reason": "very_high_uniqueness"}
    if _looks_like_text(series):
        return {"type": "text", "reason": "long_text_detected"}
    return {"type": "categorical", "reason": "categorical_or_string_variable"}

def analyze_variable_types(df):
    results = {}
    for column in df.columns:
        series = df[column]
        detection = _detect_variable_type(series, column=column)
        results[column] = {
            "dtype": str(series.dtype),
            "detected_type": detection["type"],
            "reason": detection["reason"],
            "missing": int(series.isna().sum()),
            "non_null": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True))
        }
    return results

def apply_variable_type_overrides(profile, overrides=None):
    if not overrides:
        return profile
    updated = {
        **profile,
        "variable_types": {column: dict(info) for column, info in profile.get("variable_types", {}).items()},
        "columns": [dict(info) for info in profile.get("columns", [])],
    }
    for column, selected_type in overrides.items():
        if column not in updated["variable_types"]:
            continue
        if selected_type not in {"numeric", "categorical", "identifier", "temporal"}:
            continue
        info = updated["variable_types"][column]
        actual_type = selected_type
        if selected_type == "temporal":
            auto_type = info.get("detected_type")
            actual_type = ("datetime" if auto_type == "datetime" else "time_index")
        info["auto_detected_type"] = info.get("detected_type")
        info["auto_detected_reason"] = info.get("reason")
        info["detected_type"] = actual_type
        info["reason"] = (
            "user_override_numeric"
            if selected_type == "numeric"
            else "user_override_categorical"
            if selected_type == "categorical"
            else "user_override_identifier"
            if selected_type == "identifier"
            else "user_override_temporal"
        )
        info["type_source"] = "user"
    for info in updated["variable_types"].values():
        info.setdefault("type_source", "automatic")
        info.setdefault("auto_detected_type", info.get("detected_type"))
        info.setdefault("auto_detected_reason", info.get("reason"))
    for column_info in updated["columns"]:
        column = column_info.get("name")
        if column not in updated["variable_types"]:
            continue
        variable_info = updated["variable_types"][column]
        column_info["detected_type"] = variable_info["detected_type"]
        column_info["type_reason"] = variable_info["reason"]
        column_info["type_source"] = variable_info.get("type_source", "automatic")
        if "auto_detected_type" in variable_info:
            column_info["auto_detected_type"] = variable_info["auto_detected_type"]
    type_counts = {}
    for info in updated["variable_types"].values():
        variable_type = info["detected_type"]
        type_counts[variable_type] = type_counts.get(variable_type, 0) + 1
    updated["type_counts"] = type_counts
    return updated

def analyze_profile(df):
    rows, columns = df.shape
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_usage = int(df.memory_usage(deep=True).sum())
    variable_types = (analyze_variable_types(df))
    column_info = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        non_null = int(df[column].notna().sum())
        unique = int(df[column].nunique(dropna=True))
        missing_ratio = (missing / rows if rows > 0 else 0)
        column_info.append({
            "name": column,
            "dtype": str(df[column].dtype),
            "detected_type": (variable_types[column] ["detected_type"]),
            "type_reason": (variable_types[column] ["reason"]),
            "non_null": non_null,
            "missing": missing,
            "missing_ratio": (missing_ratio),
            "unique": unique
        })
    type_counts = {}
    for info in variable_types.values():
        variable_type = (info["detected_type"])
        type_counts[variable_type] = (type_counts.get(variable_type, 0) + 1)
    return {
        "dataset": {
            "rows": rows,
            "columns": columns,
            "missing_values": (total_missing),
            "duplicate_rows": (duplicate_rows),
            "memory_usage": (memory_usage)
        },
        "columns": column_info,
        "variable_types": variable_types,
        "type_counts": type_counts
    }
