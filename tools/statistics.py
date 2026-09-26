import math
import numpy as np
import pandas as pd
from scipy import stats

def _safe_float(value):
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)

def _get_numeric_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    return [column for column, info in variable_types.items() if (column in df.columns and info.get("detected_type") == "numeric")]

def _get_categorical_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    return [column for column, info in variable_types.items() if (column in df.columns and info.get("detected_type") in categorical_types)]

def _calculate_mad(series):
    series = series.dropna()
    if len(series) == 0:
        return None
    median = series.median()
    mad = np.median(np.abs(series - median))
    return _safe_float(mad)

def _calculate_outliers(series):
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(series)
    if n == 0:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "count": 0,
            "ratio": 0.0,
            "status": "skipped",
            "reason": "no_valid_values"
        }
    if series.nunique() <= 1:
        return {
            "q1": _safe_float(series.iloc[0]),
            "q3": _safe_float(series.iloc[0]),
            "iqr": 0.0,
            "lower_bound": _safe_float(series.iloc[0]),
            "upper_bound": _safe_float(series.iloc[0]),
            "count": 0,
            "ratio": 0.0,
            "status": "skipped",
            "reason": "constant_variable"
        }
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = ((series < lower) | (series > upper))
    count = int(mask.sum())
    ratio = count / n
    return {
        "q1": _safe_float(q1),
        "q3": _safe_float(q3),
        "iqr": _safe_float(iqr),
        "lower_bound": _safe_float(lower),
        "upper_bound": _safe_float(upper),
        "count": count,
        "ratio": float(ratio),
        "status": "ok",
        "reason": None
    }

def _normality_test(series):
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(series)
    if n < 3:
        return {"test": None, "statistic": None, "p_value": None, "status": "skipped", "reason": "insufficient_samples"}
    if series.nunique() <= 1:
        return {"test": None, "statistic": None, "p_value": None, "status": "skipped", "reason": "constant_variable"}
    try:
        if n <= 5000:
            statistic, p_value = stats.shapiro(series)
            test_name = "Shapiro-Wilk"
        else:
            statistic, p_value = stats.jarque_bera(series)
            test_name = "Jarque-Bera"
        statistic = _safe_float(statistic)
        p_value = _safe_float(p_value)
        if statistic is None or p_value is None:
            return {"test": test_name, "statistic": statistic, "p_value": p_value, "status": "skipped", "reason": "non_finite_result"}
        return {"test": test_name, "statistic": statistic, "p_value": p_value, "status": "ok", "reason": None}
    except Exception as error:
        return {"test": None, "statistic": None, "p_value": None, "status": "skipped", "reason": type(error).__name__}

def analyze_numeric(df, variable_types=None):
    results = {}
    numeric_columns = _get_numeric_columns(df, variable_types)
    for column in numeric_columns:
        series = df[column].dropna()
        n = len(series)
        if n == 0:
            continue
        mean = series.mean()
        median = series.median()
        std = series.std()
        variance = series.var()
        minimum = series.min()
        maximum = series.max()
        range_value = maximum - minimum
        quantiles = series.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        q1 = quantiles.loc[0.25]
        q3 = quantiles.loc[0.75]
        iqr = q3 - q1
        skewness = series.skew()
        kurtosis = series.kurt()
        mad = _calculate_mad(series)
        if mean != 0:
            coefficient_variation = std / abs(mean)
        else:
            coefficient_variation = None
        outliers = _calculate_outliers(series)
        normality = _normality_test(series)
        results[column] = {
            "count": n,
            "missing": int(df[column].isna().sum()),
            "mean": _safe_float(mean),
            "median": _safe_float(median),
            "std": _safe_float(std),
            "variance": _safe_float(variance),
            "min": _safe_float(minimum),
            "max": _safe_float(maximum),
            "range": _safe_float(range_value),
            "quantiles": {
                "q01": _safe_float(quantiles.loc[0.01]),
                "q05": _safe_float(quantiles.loc[0.05]),
                "q10": _safe_float(quantiles.loc[0.10]),
                "q25": _safe_float(quantiles.loc[0.25]),
                "q50": _safe_float(quantiles.loc[0.50]),
                "q75": _safe_float(quantiles.loc[0.75]),
                "q90": _safe_float(quantiles.loc[0.90]),
                "q95": _safe_float(quantiles.loc[0.95]),
                "q99": _safe_float(quantiles.loc[0.99])
            },
            "iqr": _safe_float(iqr),
            "skewness": _safe_float(skewness),
            "kurtosis": _safe_float(kurtosis),
            "mad": _safe_float(mad),
            "coefficient_of_variation":
                _safe_float(coefficient_variation),
            "outliers": outliers,
            "normality_test": normality
        }
    return results

def _calculate_entropy(value_counts):
    probabilities = (value_counts / value_counts.sum())
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)

def analyze_categorical(df, variable_types=None):
    results = {}
    categorical_columns = _get_categorical_columns(df, variable_types)
    for column in categorical_columns:
        series = df[column]
        non_null = series.dropna()
        n = len(non_null)
        unique = int(non_null.nunique())
        value_counts = (non_null.value_counts(dropna=False))
        if n > 0:
            mode = value_counts.index[0]
            mode_count = int(value_counts.iloc[0])
            mode_ratio = (mode_count / n)
            entropy = _calculate_entropy(value_counts)
            if unique > 1:
                normalized_entropy = (entropy / math.log2(unique))
            else:
                normalized_entropy = 0.0
        else:
            mode = None
            mode_count = 0
            mode_ratio = 0.0
            entropy = 0.0
            normalized_entropy = 0.0
        frequencies = []
        for category, count in (value_counts.items()):
            frequencies.append({"category": str(category), "count": int(count), "ratio": float(count / n) if n > 0 else 0.0})
        results[column] = {
            "count": n,
            "missing": int(series.isna().sum()),
            "unique": unique,
            "mode": str(mode)
            if mode is not None
            else None,
            "mode_count": mode_count,
            "mode_ratio": float(mode_ratio),
            "entropy": entropy,
            "normalized_entropy":
                normalized_entropy,
            "frequencies": frequencies
        }
    return results

def analyze_statistics(df, variable_types=None):
    return {"numeric": analyze_numeric(df, variable_types), "categorical": analyze_categorical(df, variable_types)}
