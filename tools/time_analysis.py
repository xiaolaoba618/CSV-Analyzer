import numpy as np
import pandas as pd
DEFAULT_TOP_INTERVALS = 10
DEFAULT_MAX_DAILY_RESULTS = 100

def _get_datetime_columns(df, variable_types=None):
    if variable_types is None:
        return [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    datetime_columns = []
    for column, info in variable_types.items():
        if column not in df.columns:
            continue
        detected_type = info.get("detected_type")
        if detected_type == "datetime":
            datetime_columns.append(column)
    if ("DATETIME" in df.columns and "DATETIME" not in datetime_columns and pd.api.types.is_datetime64_any_dtype(df["DATETIME"])):
        datetime_columns.append("DATETIME")
    return datetime_columns

def analyze_time_range(df, datetime_column):
    series = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    if series.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "column": datetime_column}
    minimum = series.min()
    maximum = series.max()
    span = (maximum - minimum)
    return {
        "status": "ok",
        "column": datetime_column,
        "min_datetime": minimum.isoformat(),
        "max_datetime": maximum.isoformat(),
        "span_seconds": float(span.total_seconds()),
        "span_days": float(span.total_seconds() / 86400),
        "valid_count": int(series.shape[0]),
        "missing_count": int(df[datetime_column].isna().sum())
    }

def analyze_time_intervals(df, datetime_column, top_n=DEFAULT_TOP_INTERVALS):
    series = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    if len(series) < 2:
        return {"status": "skipped", "reason": "insufficient_datetime_values", "column": datetime_column}
    # 计算时间间隔前先按时间排序。
    series = series.sort_values()
    intervals = (series.diff().dropna())
    intervals_seconds = (intervals.dt.total_seconds())
    if intervals_seconds.empty:
        return {"status": "skipped", "reason": "unable_to_calculate_intervals", "column": datetime_column}
    interval_counts = (intervals_seconds.value_counts().sort_values(ascending=False))
    total_intervals = int(len(intervals_seconds))
    interval_results = []
    for seconds, count in (interval_counts.head(top_n).items()):
        count = int(count)
        interval_results.append({
            "interval_seconds": float(seconds),
            "interval_minutes": float(seconds / 60),
            "interval_hours": float(seconds / 3600),
            "count": count,
            "ratio": float(count / total_intervals)
        })
    median_seconds = float(intervals_seconds.median())
    mean_seconds = float(intervals_seconds.mean())
    minimum_seconds = float(intervals_seconds.min())
    maximum_seconds = float(intervals_seconds.max())
    return {
        "status": "ok",
        "column": datetime_column,
        "total_intervals": total_intervals,
        "mean_interval_seconds": mean_seconds,
        "median_interval_seconds": median_seconds,
        "min_interval_seconds": minimum_seconds,
        "max_interval_seconds": maximum_seconds,
        "intervals": interval_results
    }

def analyze_hourly_distribution(df, datetime_column):
    series = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    if series.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "column": datetime_column}
    counts = (series.dt.hour.value_counts().sort_index())
    results = []
    total = int(len(series))
    for hour in range(24):
        count = int(counts.get(hour, 0))
        results.append({"hour": hour, "count": count, "ratio": float(count / total)})
    return {"status": "ok", "column": datetime_column, "results": results}

def analyze_daily_distribution(df, datetime_column, max_results=DEFAULT_MAX_DAILY_RESULTS):
    series = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    if series.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "column": datetime_column}
    date_counts = (series.dt.date.value_counts().sort_index())
    results = []
    for date_value, count in date_counts.items():
        results.append({"date": str(date_value), "count": int(count)})
    total_days = len(results)
    if total_days > max_results:
        display_results = results[-max_results:]
        truncated = True
    else:
        display_results = results
        truncated = False
    return {
        "status": "ok",
        "column": datetime_column,
        "unique_dates": total_days,
        "results": display_results,
        "displayed_results": len(display_results),
        "truncated": truncated,
        "max_results": max_results
    }

def analyze_weekday_distribution(df, datetime_column):
    series = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    if series.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "column": datetime_column}
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = (series.dt.dayofweek.value_counts().sort_index())
    total = int(len(series))
    results = []
    for weekday in range(7):
        count = int(counts.get(weekday, 0))
        results.append({"weekday": weekday_names[weekday], "weekday_index": weekday, "count": count, "ratio": float(count / total)})
    return {"status": "ok", "column": datetime_column, "results": results}

def analyze_time_analysis(df, variable_types=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    datetime_columns = (_get_datetime_columns(df, variable_types))
    results = {"status": "ok", "datetime_columns": datetime_columns, "analyses": {}}
    if not datetime_columns:
        results["status"] = "skipped"
        results["reason"] = ("no_datetime_variable")
        return results
    for column in datetime_columns:
        results["analyses"][column] = {
            "time_range": (analyze_time_range(df, column)),
            "time_intervals": (analyze_time_intervals(df, column)),
            "hourly_distribution": (analyze_hourly_distribution(df, column)),
            "daily_distribution": (analyze_daily_distribution(df, column)),
            "weekday_distribution": (analyze_weekday_distribution(df, column))
        }
    return results
