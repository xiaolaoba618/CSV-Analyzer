from pathlib import Path
import numpy as np
import pandas as pd
DEFAULT_MIN_OBSERVATIONS = 3
DEFAULT_ANOMALY_WINDOW = 7
DEFAULT_ANOMALY_THRESHOLD = 3.5

def _get_time_index_columns(df, variable_types=None):
    if variable_types is None:
        return []
    return [column for column, info in variable_types.items() if column in df.columns and info.get("detected_type") == "time_index"]

def _prepare_time_index_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")

def _aggregate_time_index_series(df, time_column, numeric_columns):
    if not numeric_columns:
        return {"status": "skipped", "reason": "no_numeric_variables"}
    work = df[[time_column] + numeric_columns].copy()
    work[time_column] = pd.to_numeric(work[time_column], errors="coerce")
    work = work.dropna(subset=[time_column])
    if work.empty:
        return {"status": "skipped", "reason": "no_valid_time_index_values"}
    grouped = work.groupby(time_column, sort=True)
    variables = {}
    for column in numeric_columns:
        values = grouped[column].mean().dropna()
        variables[column] = {
            "status": "ok",
            "results": [{"time_index": float(index), "mean": float(value)} for index, value in values.items()]
        }
    return {"status": "ok", "frequency": "time_index", "time_column": time_column, "variables": variables}

def _get_datetime_columns(df, variable_types=None):
    if variable_types is None:
        datetime_columns = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    else:
        datetime_columns = []
        for column, info in variable_types.items():
            if column not in df.columns:
                continue
            detected_type = info.get("detected_type")
            if detected_type == "datetime":
                datetime_columns.append(column)
    # 纳入预处理生成的 DATETIME 列。
    if ("DATETIME" in df.columns and "DATETIME" not in datetime_columns and pd.api.types.is_datetime64_any_dtype(df["DATETIME"])):
        datetime_columns.append("DATETIME")
    return datetime_columns

def _select_datetime_column(df, datetime_columns, requested_column=None):
    if not datetime_columns:
        return {"status": "skipped", "reason": "no_datetime_variable", "datetime_column": None}
    # 用户显式选择优先。
    if requested_column is not None:
        if requested_column in datetime_columns:
            return {"status": "ok", "datetime_column": requested_column, "selection_reason": "user_selected"}
        return {
            "status": "skipped",
            "reason": "invalid_datetime_column",
            "requested_column": requested_column,
            "datetime_columns": datetime_columns
        }
    # 优先使用预处理合并后的 DATE + TIME 列。
    if "DATETIME" in datetime_columns:
        return {"status": "ok", "datetime_column": "DATETIME", "selection_reason": "combined_datetime"}
    # 只有一个日期时间变量时直接使用。
    if len(datetime_columns) == 1:
        return {"status": "ok", "datetime_column": datetime_columns[0], "selection_reason": "single_datetime_variable"}
    # 多个日期时间变量时保持歧义，不强行选择。
    return {"status": "skipped", "reason": "multiple_datetime_columns", "datetime_columns": datetime_columns}

def _get_numeric_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    numeric_columns = []
    for column, info in variable_types.items():
        if column not in df.columns:
            continue
        detected_type = info.get("detected_type")
        if detected_type == "numeric":
            numeric_columns.append(column)
    return numeric_columns

def _prepare_datetime_series(df, datetime_column):
    if datetime_column not in df.columns:
        raise KeyError(
            f"Datetime column not found in dataframe: {datetime_column}"
        )
    series = pd.to_datetime(df[datetime_column], errors="coerce")
    return series

def _detect_time_frequency(datetime_series, tolerance=0.20):
    series = pd.to_datetime(datetime_series, errors="coerce").dropna()
    if len(series) < 2:
        return {"status": "skipped", "reason": "insufficient_datetime_values"}
    series = series.sort_values()
    intervals = series.diff().dropna()
    intervals_seconds = intervals.dt.total_seconds()
    intervals_seconds = intervals_seconds[intervals_seconds > 0]
    if intervals_seconds.empty:
        return {"status": "skipped", "reason": "no_positive_time_intervals"}
    median_seconds = float(intervals_seconds.median())
    mean_seconds = float(intervals_seconds.mean())
    minimum_seconds = float(intervals_seconds.min())
    maximum_seconds = float(intervals_seconds.max())
    frequency = "irregular"
    reference_intervals = {
        "minute": 60,
        "hourly": 3600,
        "daily": 86400,
        "weekly": 604800,
        "monthly": 30.4375 * 86400,
        "yearly": 365.25 * 86400
    }
    for name, seconds in reference_intervals.items():
        relative_difference = abs(median_seconds - seconds) / seconds
        if relative_difference <= tolerance:
            frequency = name
            break
    return {
        "status": "ok",
        "detected": frequency,
        "median_interval_seconds": median_seconds,
        "mean_interval_seconds": mean_seconds,
        "minimum_interval_seconds": minimum_seconds,
        "maximum_interval_seconds": maximum_seconds,
        "total_intervals": int(len(intervals_seconds))
    }

def _get_resample_rule(frequency):
    rules = {"minute": "min", "hourly": "h", "daily": "D", "weekly": "W", "monthly": "ME", "yearly": "YE"}
    return rules.get(frequency)

def _aggregate_time_series(df, datetime_column, numeric_columns, frequency):
    if not numeric_columns:
        return {"status": "skipped", "reason": "no_numeric_variables"}
    rule = _get_resample_rule(frequency)
    if rule is None:
        return {"status": "skipped", "reason": "irregular_frequency"}
    work_df = df[[datetime_column] + numeric_columns].copy()
    work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
    work_df = work_df.dropna(subset=[datetime_column])
    if work_df.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values"}
    work_df = work_df.sort_values(datetime_column)
    work_df = work_df.set_index(datetime_column)
    results = {}
    for column in numeric_columns:
        try:
            series = pd.to_numeric(work_df[column], errors="coerce")
            aggregated = pd.DataFrame({
                "mean": series.resample(rule).mean(),
                "median": series.resample(rule).median(),
                "min": series.resample(rule).min(),
                "max": series.resample(rule).max(),
                "count": series.resample(rule).count()
            })
            aggregated = aggregated.dropna(how="all")
            records = []
            for timestamp, row in aggregated.iterrows():
                records.append({
                    "datetime": timestamp.isoformat(),
                    "mean": (float(row["mean"]) if pd.notna(row["mean"]) else None),
                    "median": (float(row["median"]) if pd.notna(row["median"]) else None),
                    "min": (float(row["min"]) if pd.notna(row["min"]) else None),
                    "max": (float(row["max"]) if pd.notna(row["max"]) else None),
                    "count": int(row["count"])
                })
            results[column] = {"status": "ok", "frequency": frequency, "rule": rule, "results": records}
        except Exception as error:
            results[column] = {"status": "skipped", "reason": type(error).__name__}
    return {"status": "ok", "frequency": frequency, "variables": results}

def _prepare_irregular_time_series(df, datetime_column, numeric_columns):
    if not numeric_columns:
        return {"status": "skipped", "reason": "no_numeric_variables", "variables": {}}
    work_df = df[[datetime_column] + numeric_columns].copy()
    work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
    work_df = work_df.dropna(subset=[datetime_column])
    if work_df.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "variables": {}}
    work_df = work_df.sort_values(datetime_column)
    results = {}
    for column in numeric_columns:
        try:
            series = pd.to_numeric(work_df[column], errors="coerce")
            valid = pd.DataFrame({"datetime": work_df[datetime_column], "value": series})
            valid = valid.dropna(subset=["datetime", "value"])
            if valid.empty:
                results[column] = {"status": "skipped", "reason": "no_valid_numeric_values"}
                continue
            records = []
            for _, row in valid.iterrows():
                records.append({"datetime": row["datetime"].isoformat(), "value": float(row["value"])})
            results[column] = {"status": "ok", "frequency": "irregular", "results": records}
        except Exception as error:
            results[column] = {"status": "skipped", "reason": type(error).__name__}
    return {"status": "ok", "frequency": "irregular", "variables": results}

def _calculate_linear_trend(series, min_observations=DEFAULT_MIN_OBSERVATIONS):
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_series) < min_observations:
        return {"status": "skipped", "reason": "insufficient_observations", "count": int(len(clean_series))}
    values = clean_series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        values = values[np.isfinite(values)]
    if len(values) < min_observations:
        return {"status": "skipped", "reason": "insufficient_finite_observations", "count": int(len(values))}
    x = np.arange(len(values), dtype=float)
    if np.allclose(values, values[0]):
        return {
            "status": "ok",
            "direction": "stable",
            "slope": 0.0,
            "intercept": float(values[0]),
            "r_squared": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "count": int(len(values))
        }
    try:
        slope, intercept = np.polyfit(x, values, 1)
        predicted = (slope * x + intercept)
        residuals = values - predicted
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        if ss_tot > 0:
            r_squared = (1.0 - ss_res / ss_tot)
        else:
            r_squared = 0.0
        first_value = float(values[0])
        last_value = float(values[-1])
        change = (last_value - first_value)
        if abs(first_value) > 1e-12:
            change_percent = (change / abs(first_value) * 100)
        else:
            change_percent = None
        slope_tolerance = (max(abs(values.mean()), 1.0) * 1e-10)
        if slope > slope_tolerance:
            direction = "increasing"
        elif slope < -slope_tolerance:
            direction = "decreasing"
        else:
            direction = "stable"
        return {
            "status": "ok",
            "direction": direction,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(max(0.0, min(1.0, r_squared))),
            "first_value": first_value,
            "last_value": last_value,
            "change": float(change),
            "change_percent": (float(change_percent) if change_percent is not None else None),
            "count": int(len(values))
        }
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__}

def _detect_anomalies(series, window=DEFAULT_ANOMALY_WINDOW, threshold=DEFAULT_ANOMALY_THRESHOLD):
    clean_series = pd.to_numeric(series, errors="coerce")
    if clean_series.dropna().empty:
        return {"status": "skipped", "reason": "no_numeric_values"}
    if len(clean_series.dropna()) < window:
        return {"status": "skipped", "reason": "insufficient_observations"}
    rolling_median = (clean_series.rolling(window=window, center=True, min_periods=max(3, window // 2)).median())
    absolute_deviation = (clean_series - rolling_median).abs()
    rolling_mad = (absolute_deviation.rolling(window=window, center=True, min_periods=max(3, window // 2)).median())
    robust_z = (0.6745 * (clean_series - rolling_median) / rolling_mad.replace(0, np.nan))
    anomaly_mask = (robust_z.abs() > threshold)
    anomaly_indices = (robust_z.index[anomaly_mask.fillna(False)])
    anomalies = []
    for index in anomaly_indices:
        value = clean_series.loc[index]
        score = robust_z.loc[index]
        if pd.isna(value) or pd.isna(score):
            continue
        anomalies.append({
            "index": (int(index) if isinstance(index, (int, np.integer)) else str(index)),
            "value": float(value),
            "robust_z_score": float(score)
        })
    return {
        "status": "ok",
        "method": "rolling_mad",
        "window": int(window),
        "threshold": float(threshold),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies
    }

def _analyze_single_variable(
    series,
    min_observations=DEFAULT_MIN_OBSERVATIONS,
    anomaly_window=DEFAULT_ANOMALY_WINDOW,
    anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
):
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_series = numeric_series.dropna()
    if len(valid_series) < min_observations:
        return {"status": "skipped", "reason": "insufficient_observations", "count": int(len(valid_series))}
    trend = _calculate_linear_trend(numeric_series, min_observations=min_observations)
    anomalies = _detect_anomalies(numeric_series, window=anomaly_window, threshold=anomaly_threshold)
    return {
        "status": "ok",
        "count": int(len(valid_series)),
        "missing_count": int(numeric_series.isna().sum()),
        "mean": float(valid_series.mean()),
        "median": float(valid_series.median()),
        "std": float(valid_series.std()) if len(valid_series) > 1 else 0.0,
        "trend": trend,
        "anomalies": anomalies
    }

def _analyze_original_series(
    df,
    datetime_column,
    numeric_columns,
    min_observations=DEFAULT_MIN_OBSERVATIONS,
    anomaly_window=DEFAULT_ANOMALY_WINDOW,
    anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
):
    if not numeric_columns:
        return {"status": "skipped", "reason": "no_numeric_variables", "variables": {}}
    work_df = df[[datetime_column] + numeric_columns].copy()
    work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
    work_df = work_df.dropna(subset=[datetime_column])
    if work_df.empty:
        return {"status": "skipped", "reason": "no_valid_datetime_values", "variables": {}}
    work_df = work_df.sort_values(datetime_column)
    results = {}
    for column in numeric_columns:
        try:
            results[column] = _analyze_single_variable(
                work_df[column],
                min_observations=min_observations,
                anomaly_window=anomaly_window,
                anomaly_threshold=anomaly_threshold
            )
        except Exception as error:
            results[column] = {"status": "skipped", "reason": type(error).__name__}
    return {"status": "ok", "datetime_column": datetime_column, "variables": results}

def analyze_time_series(
    df,
    variable_types=None,
    datetime_column=None,
    min_observations=DEFAULT_MIN_OBSERVATIONS,
    anomaly_window=DEFAULT_ANOMALY_WINDOW,
    anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    datetime_columns = _get_datetime_columns(df, variable_types)
    # 有序时间变量单独处理，避免被当作普通数值变量。
    time_index_columns = _get_time_index_columns(df, variable_types)
    if not datetime_columns and time_index_columns:
        selected_time_column = time_index_columns[0]
        numeric_columns = _get_numeric_columns(df, variable_types)
        numeric_columns = [c for c in numeric_columns if c != selected_time_column]
        aggregation = _aggregate_time_index_series(df, selected_time_column, numeric_columns)
        if aggregation.get("status") != "ok":
            return {
                "status": "skipped",
                "reason": aggregation.get("reason", "time_index_analysis_failed"),
                "time_index_columns": time_index_columns,
                "time_column": selected_time_column,
                "numeric_columns": numeric_columns,
                "analyses": {}
            }
        return {
            "status": "ok",
            "time_type": "time_index",
            "datetime_columns": [],
            "time_index_columns": time_index_columns,
            "time_column": selected_time_column,
            "numeric_columns": numeric_columns,
            "frequency": {"status": "ok", "detected": "time_index"},
            "aggregation": aggregation,
            "analyses": {}
        }
    if not datetime_columns:
        return {
            "status": "skipped",
            "reason": "no_datetime_or_time_index_variable",
            "datetime_columns": [],
            "time_index_columns": time_index_columns,
            "numeric_columns": [],
            "analyses": {}
        }
    datetime_selection = _select_datetime_column(df, datetime_columns, requested_column=datetime_column)
    if datetime_selection.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": datetime_selection.get("reason", "datetime_selection_failed"),
            "requested_column": datetime_column,
            "datetime_columns": datetime_columns,
            "numeric_columns": [],
            "analyses": {}
        }
    selected_datetime_column = datetime_selection["datetime_column"]
    numeric_columns = _get_numeric_columns(df, variable_types)
    # 日期时间列不作为普通数值变量分析。
    numeric_columns = [column for column in numeric_columns if column != selected_datetime_column]
    if not numeric_columns:
        return {
            "status": "skipped",
            "reason": "no_numeric_variables",
            "datetime_columns": datetime_columns,
            "datetime_column": selected_datetime_column,
            "numeric_columns": [],
            "analyses": {}
        }
    datetime_series = _prepare_datetime_series(df, selected_datetime_column)
    valid_datetime_count = int(datetime_series.notna().sum())
    if valid_datetime_count < min_observations:
        return {
            "status": "skipped",
            "reason": "insufficient_datetime_values",
            "datetime_columns": datetime_columns,
            "datetime_column": selected_datetime_column,
            "numeric_columns": numeric_columns,
            "valid_datetime_count": valid_datetime_count,
            "analyses": {}
        }
    time_frequency = _detect_time_frequency(datetime_series)
    time_range = None
    valid_dates = datetime_series.dropna()
    if not valid_dates.empty:
        minimum = valid_dates.min()
        maximum = valid_dates.max()
        span = maximum - minimum
        time_range = {
            "min_datetime": minimum.isoformat(),
            "max_datetime": maximum.isoformat(),
            "span_seconds": float(span.total_seconds()),
            "span_days": float(span.total_seconds() / 86400),
            "valid_count": int(len(valid_dates)),
            "missing_count": int(datetime_series.isna().sum())
        }
    original_series_analysis = _analyze_original_series(
        df,
        selected_datetime_column,
        numeric_columns,
        min_observations=min_observations,
        anomaly_window=anomaly_window,
        anomaly_threshold=anomaly_threshold
    )
    if (time_frequency.get("status") == "ok" and time_frequency.get("detected") != "irregular"):
        aggregation = _aggregate_time_series(df, selected_datetime_column, numeric_columns, time_frequency["detected"])
    elif time_frequency.get("detected") == "irregular":
        aggregation = _prepare_irregular_time_series(df, selected_datetime_column, numeric_columns)
    else:
        aggregation = {
            "status": "skipped",
            "reason": "unable_to_detect_frequency",
            "frequency": time_frequency.get("detected"),
            "variables": {}
        }
    return {
        "status": "ok",
        "datetime_columns": datetime_columns,
        "datetime_column": selected_datetime_column,
        "datetime_selection_reason": datetime_selection.get("selection_reason"),
        "numeric_columns": numeric_columns,
        "time_range": time_range,
        "frequency": time_frequency,
        "original_series": original_series_analysis,
        "aggregation": aggregation
    }
