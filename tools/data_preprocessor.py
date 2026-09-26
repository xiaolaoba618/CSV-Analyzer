from pathlib import Path
import numpy as np
import pandas as pd
NUMERIC_MISSING_TOKENS = {"", "-", "--", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "."}
DEFAULT_DATETIME_COLUMN = "DATETIME"

def _clean_string_series(series):
    return series.astype(str).str.strip()

def _replace_numeric_missing_tokens(series):
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace(list(NUMERIC_MISSING_TOKENS), np.nan)
    return cleaned

def _safe_numeric_conversion(series):
    try:
        cleaned = _replace_numeric_missing_tokens(series)
        converted = pd.to_numeric(cleaned, errors="coerce")
        return converted
    except Exception:
        return None

def preprocess_numeric_column(df, column):
    if column not in df.columns:
        return {"status": "skipped", "reason": "column_not_found"}
    try:
        original_series = df[column]
        converted = _safe_numeric_conversion(original_series)
        if converted is None:
            return {"status": "skipped", "reason": "numeric_conversion_failed"}
        original_missing = int(original_series.isna().sum())
        converted_missing = int(converted.isna().sum())
        newly_missing = (converted_missing - original_missing)
        df[column] = converted
        return {
            "status": "ok",
            "column": column,
            "original_dtype": str(original_series.dtype),
            "new_dtype": str(converted.dtype),
            "original_missing": (original_missing),
            "new_missing": (converted_missing),
            "newly_missing": max(0, newly_missing)
        }
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__, "column": column}

def preprocess_numeric_variables(df, variable_types):
    results = {}
    for column, info in variable_types.items():
        detected_type = info.get("detected_type")
        if detected_type != "numeric":
            continue
        results[column] = (preprocess_numeric_column(df, column))
    return results

def preprocess_datetime_column(df, column):
    if column not in df.columns:
        return {"status": "skipped", "reason": "column_not_found"}
    try:
        original_series = df[column]
        converted = pd.to_datetime(original_series, errors="coerce", format="mixed")
        original_missing = int(original_series.isna().sum())
        converted_missing = int(converted.isna().sum())
        newly_missing = (converted_missing - original_missing)
        df[column] = converted
        return {
            "status": "ok",
            "column": column,
            "original_dtype": str(original_series.dtype),
            "new_dtype": str(converted.dtype),
            "original_missing": (original_missing),
            "new_missing": (converted_missing),
            "newly_missing": max(0, newly_missing)
        }
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__, "column": column}

def preprocess_datetime_variables(df, variable_types):
    results = {}
    for column, info in variable_types.items():
        detected_type = info.get("detected_type")
        if detected_type != "datetime":
            continue
        results[column] = (preprocess_datetime_column(df, column))
    return results

def _convert_numeric_time(value):
    if pd.isna(value):
        return pd.NaT
    try:
        value = float(value)
        if not np.isfinite(value):
            return pd.NaT
        value = int(round(value))
        if value < 0 or value > 2359:
            return pd.NaT
        hour = value // 100
        minute = value % 100
        if hour > 23 or minute > 59:
            return pd.NaT
        return pd.Timestamp(year=1900, month=1, day=1, hour=hour, minute=minute)
    except Exception:
        return pd.NaT

def _convert_string_time(value):
    if pd.isna(value):
        return pd.NaT
    value = str(value).strip()
    if value in NUMERIC_MISSING_TOKENS:
        return pd.NaT
    try:
        return pd.to_datetime(value, format="%H:%M:%S")
    except Exception:
        pass
    try:
        return pd.to_datetime(value, format="%H:%M")
    except Exception:
        pass
    try:
        numeric_value = float(value)
        return _convert_numeric_time(numeric_value)
    except Exception:
        pass
    return pd.NaT

def _convert_time_series(series):
    results = []
    for value in series:
        if pd.api.types.is_numeric_dtype(series):
            converted = (_convert_numeric_time(value))
        else:
            converted = (_convert_string_time(value))
        results.append(converted)
    return pd.Series(results, index=series.index, dtype="datetime64[ns]")

def preprocess_time_column(df, column):
    if column not in df.columns:
        return {"status": "skipped", "reason": "column_not_found"}
    try:
        original_series = df[column]
        converted = _convert_time_series(original_series)
        original_missing = int(original_series.isna().sum())
        converted_missing = int(converted.isna().sum())
        newly_missing = (converted_missing - original_missing)
        df[column] = converted
        return {
            "status": "ok",
            "column": column,
            "original_dtype": str(original_series.dtype),
            "new_dtype": str(converted.dtype),
            "original_missing": (original_missing),
            "new_missing": (converted_missing),
            "newly_missing": max(0, newly_missing)
        }
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__, "column": column}

def preprocess_time_variables(df, variable_types):
    results = {}
    for column, info in variable_types.items():
        detected_type = info.get("detected_type")
        if detected_type != "time":
            continue
        results[column] = (preprocess_time_column(df, column))
    return results

def combine_date_and_time(df, variable_types, datetime_column=DEFAULT_DATETIME_COLUMN):
    date_columns = [column for column, info in variable_types.items() if (info.get("detected_type") == "datetime" and column in df.columns)]
    time_columns = [column for column, info in variable_types.items() if (info.get("detected_type") == "time" and column in df.columns)]
    # 仅在存在唯一日期时间变量和唯一时间变量时合并。
    if len(date_columns) != 1 or len(time_columns) != 1:
        return {
            "status": "skipped",
            "reason": "datetime_time_pair_not_unique",
            "datetime_columns": date_columns,
            "time_columns": time_columns,
            "datetime_count": len(date_columns),
            "time_count": len(time_columns)
        }
    date_column = date_columns[0]
    time_column = time_columns[0]
    try:
        date_series = pd.to_datetime(df[date_column], errors="coerce")
        time_series = pd.to_datetime(df[time_column], errors="coerce")
        # 按行合并日期与时间。
        combined = (date_series.dt.normalize() + (time_series - time_series.dt.normalize()))
        valid_count = int(combined.notna().sum())
        if valid_count == 0:
            return {
                "status": "skipped",
                "reason": "unable_to_combine_date_and_time",
                "date_column": date_column,
                "time_column": time_column
            }
        df[datetime_column] = combined
        return {
            "status": "ok",
            "column": datetime_column,
            "date_column": date_column,
            "time_column": time_column,
            "valid_count": valid_count,
            "missing": int(combined.isna().sum())
        }
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__, "date_column": date_column, "time_column": time_column}

def preprocess_dataframe(df, variable_types, create_combined_datetime=True):
    processed_df = df.copy()
    transformations = {"numeric": {}, "datetime": {}, "time": {}, "combined_datetime": None}
    transformations["numeric"] = (preprocess_numeric_variables(processed_df, variable_types))
    transformations["datetime"] = (preprocess_datetime_variables(processed_df, variable_types))
    transformations["time"] = (preprocess_time_variables(processed_df, variable_types))
    if create_combined_datetime:
        transformations["combined_datetime"] = combine_date_and_time(processed_df, variable_types)
    return processed_df, transformations

def summarize_preprocessing(transformations):
    summary = {
        "numeric_processed": 0,
        "datetime_processed": 0,
        "time_processed": 0,
        "combined_datetime_created": False,
        "new_missing_values": 0,
        "skipped_operations": []
    }
    numeric_results = transformations.get("numeric", {})
    for column, result in numeric_results.items():
        if result.get("status") == "ok":
            summary["numeric_processed"] += 1
            summary["new_missing_values"] += result.get("newly_missing", 0)
        else:
            summary["skipped_operations"].append({"operation": "numeric", "column": column, "reason": result.get("reason")})
    datetime_results = transformations.get("datetime", {})
    for column, result in datetime_results.items():
        if result.get("status") == "ok":
            summary["datetime_processed"] += 1
            summary["new_missing_values"] += result.get("newly_missing", 0)
        else:
            summary["skipped_operations"].append({"operation": "datetime", "column": column, "reason": result.get("reason")})
    time_results = transformations.get("time", {})
    for column, result in time_results.items():
        if result.get("status") == "ok":
            summary["time_processed"] += 1
            summary["new_missing_values"] += result.get("newly_missing", 0)
        else:
            summary["skipped_operations"].append({"operation": "time", "column": column, "reason": result.get("reason")})
    combined_result = transformations.get("combined_datetime")
    if (combined_result and combined_result.get("status") == "ok"):
        summary["combined_datetime_created"] = True
    elif combined_result:
        summary["skipped_operations"].append({"operation": "combine_datetime", "reason": combined_result.get("reason")})
    return summary

def update_profile_after_preprocessing(profile, df):
    updated_profile = profile.copy()
    if "dataset" in updated_profile:
        updated_profile["dataset"] = (updated_profile["dataset"].copy())
        updated_profile["dataset"]["rows"] = (int(df.shape[0]))
        updated_profile["dataset"]["columns"] = (int(df.shape[1]))
        updated_profile["dataset"]["missing_values"] = (int(df.isna().sum().sum()))
        updated_profile["dataset"]["duplicate_rows"] = (int(df.duplicated().sum()))
    updated_columns = []
    for column_info in updated_profile.get("columns", []):
        column_info = column_info.copy()
        column_name = column_info["name"]
        if column_name not in df.columns:
            updated_columns.append(column_info)
            continue
        series = df[column_name]
        column_info["dtype"] = (str(series.dtype))
        column_info["non_null"] = (int(series.notna().sum()))
        column_info["missing"] = (int(series.isna().sum()))
        column_info["missing_ratio"] = (float(series.isna().mean()))
        column_info["unique"] = (int(series.nunique(dropna=True)))
        updated_columns.append(column_info)
    updated_profile["columns"] = (updated_columns)
    return updated_profile
