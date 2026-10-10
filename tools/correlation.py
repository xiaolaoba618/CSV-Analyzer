import numpy as np
import pandas as pd
from scipy import stats
STRONG_CORRELATION_THRESHOLD = 0.7
MODERATE_CORRELATION_THRESHOLD = 0.4
# 限制 Kendall 相关分析的数据规模。
KENDALL_MAX_SAMPLES = 50000

def _get_numeric_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    return [column for column, info in variable_types.items() if (column in df.columns and info.get("detected_type") == "numeric")]

def _safe_float(value):
    if pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value

def _correlation_level(correlation):
    if correlation is None:
        return "unknown"
    absolute_value = abs(correlation)
    if absolute_value >= STRONG_CORRELATION_THRESHOLD:
        return "strong"
    if absolute_value >= MODERATE_CORRELATION_THRESHOLD:
        return "moderate"
    return "weak"

def _get_safe_series(df, column):
    try:
        result = df[column]
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return result
    except Exception:
        return None

def _prepare_pair(df, column_x, column_y):
    try:
        x = _get_safe_series(df, column_x)
        y = _get_safe_series(df, column_y)
        if x is None or y is None:
            return None, "column_access_failed"
        pair = pd.concat([x, y], axis=1)
        pair.columns = ["x", "y"]
        pair = pair.replace([np.inf, -np.inf], np.nan)
        pair = pair.dropna()
        if len(pair) < 3:
            return None, "insufficient_samples"
        x = pair["x"]
        y = pair["y"]
        if x.nunique() <= 1:
            return None, "constant_variable_x"
        if y.nunique() <= 1:
            return None, "constant_variable_y"
        return pair, None
    except Exception as error:
        return None, type(error).__name__

def _calculate_correlation(df, column_x, column_y, method):
    pair, reason = _prepare_pair(df, column_x, column_y)
    if pair is None:
        return {"status": "skipped", "reason": reason, "correlation": None, "p_value": None, "sample_size": 0}
    x = pair["x"]
    y = pair["y"]
    try:
        if method == "pearson":
            correlation, p_value = (stats.pearsonr(x, y))
        elif method == "spearman":
            correlation, p_value = (stats.spearmanr(x, y))
        elif method == "kendall":
            if len(pair) > KENDALL_MAX_SAMPLES:
                return {
                    "status": "skipped",
                    "reason": ("sample_size_exceeds_kendall_limit"),
                    "correlation": None,
                    "p_value": None,
                    "sample_size": len(pair)
                }
            correlation, p_value = (stats.kendalltau(x, y))
        else:
            return {"status": "skipped", "reason": "unknown_method", "correlation": None, "p_value": None, "sample_size": len(pair)}
        correlation = _safe_float(correlation)
        p_value = _safe_float(p_value)
        if correlation is None:
            return {
                "status": "skipped",
                "reason": "non_finite_correlation",
                "correlation": None,
                "p_value": p_value,
                "sample_size": len(pair)
            }
        return {"status": "ok", "reason": None, "correlation": correlation, "p_value": p_value, "sample_size": len(pair)}
    except Exception as error:
        return {"status": "skipped", "reason": type(error).__name__, "correlation": None, "p_value": None, "sample_size": len(pair)}

def _initialize_result_matrix(columns):
    return pd.DataFrame(np.nan, index=columns, columns=columns)

def _calculate_method(df, method, variable_types=None):
    columns = _get_numeric_columns(df, variable_types)
    correlation = _initialize_result_matrix(columns)
    p_values = _initialize_result_matrix(columns)
    sample_sizes = _initialize_result_matrix(columns)
    skipped_pairs = []
    for i, column_x in enumerate(columns):
        for j, column_y in enumerate(columns):
            if j < i:
                continue
            if column_x == column_y:
                series = _get_safe_series(df, column_x)
                if series is None:
                    skipped_pairs.append({"variable_1": column_x, "variable_2": column_y, "reason": "column_access_failed"})
                    continue
                valid = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) >= 2:
                    sample_sizes.loc[column_x, column_y] = len(valid)
                if valid.nunique() > 1:
                    correlation.loc[column_x, column_y] = 1.0
                    p_values.loc[column_x, column_y] = 0.0
                continue
            result = _calculate_correlation(df, column_x, column_y, method)
            if result["status"] == "ok":
                correlation.loc[column_x, column_y] = result["correlation"]
                correlation.loc[column_y, column_x] = result["correlation"]
                p_values.loc[column_x, column_y] = result["p_value"]
                p_values.loc[column_y, column_x] = result["p_value"]
                sample_sizes.loc[column_x, column_y] = result["sample_size"]
                sample_sizes.loc[column_y, column_x] = result["sample_size"]
            else:
                skipped_pairs.append({
                    "variable_1": column_x,
                    "variable_2": column_y,
                    "reason": result["reason"],
                    "sample_size": result["sample_size"]
                })
    return {
        "status": "ok",
        "reason": None,
        "correlation": correlation,
        "p_values": p_values,
        "sample_sizes": sample_sizes,
        "skipped_pairs": skipped_pairs
    }

def calculate_pearson(df, variable_types=None):
    return _calculate_method(df, method="pearson", variable_types=variable_types)

def calculate_spearman(df, variable_types=None):
    return _calculate_method(df, method="spearman", variable_types=variable_types)

def calculate_kendall(df, variable_types=None):
    return _calculate_method(df, method="kendall", variable_types=variable_types)

def find_important_pairs(correlation_result, method="spearman", threshold=STRONG_CORRELATION_THRESHOLD):
    correlation_matrix = correlation_result["correlation"]
    p_value_matrix = correlation_result["p_values"]
    sample_size_matrix = correlation_result["sample_sizes"]
    columns = correlation_matrix.columns
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column_x = columns[i]
            column_y = columns[j]
            value = correlation_matrix.loc[column_x, column_y]
            if pd.isna(value):
                continue
            if abs(value) < threshold:
                continue
            p_value = p_value_matrix.loc[column_x, column_y]
            sample_size = sample_size_matrix.loc[column_x, column_y]
            if pd.isna(sample_size):
                continue
            direction = ("positive" if value > 0 else "negative")
            pairs.append({
                "variable_1": column_x,
                "variable_2": column_y,
                "method": method,
                "correlation": _safe_float(value),
                "p_value": _safe_float(p_value),
                "sample_size": int(sample_size),
                "direction": direction,
                "strength": _correlation_level(value)
            })
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return pairs

def analyze_correlation(df, variable_types=None):
    numeric_columns = _get_numeric_columns(df, variable_types)
    if len(numeric_columns) < 2:
        empty_result = {
            "status": "skipped",
            "reason": "insufficient_numeric_variables",
            "correlation": pd.DataFrame(),
            "p_values": pd.DataFrame(),
            "sample_sizes": pd.DataFrame(),
            "skipped_pairs": []
        }
        return {
            "status": "skipped",
            "reason": "insufficient_numeric_variables",
            "numeric_columns": numeric_columns,
            "pearson": empty_result,
            "spearman": empty_result,
            "kendall": empty_result,
            "important_pairs": {"pearson": [], "spearman": [], "kendall": []}
        }
    pearson = calculate_pearson(df, variable_types)
    spearman = calculate_spearman(df, variable_types)
    if len(df) > KENDALL_MAX_SAMPLES:
        kendall = {
            "status": "skipped",
            "reason": ("dataset_size_exceeds_kendall_limit"),
            "correlation": pd.DataFrame(),
            "p_values": pd.DataFrame(),
            "sample_sizes": pd.DataFrame(),
            "skipped_pairs": []
        }
        kendall_important_pairs = []
    else:
        kendall = calculate_kendall(df, variable_types)
        kendall_important_pairs = (find_important_pairs(kendall, method="kendall"))
    important_pairs = {
        "pearson": find_important_pairs(pearson, method="pearson"),
        "spearman": find_important_pairs(spearman, method="spearman"),
        "kendall": kendall_important_pairs
    }
    return {
        "status": "ok",
        "reason": None,
        "numeric_columns": numeric_columns,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "important_pairs": important_pairs
    }
