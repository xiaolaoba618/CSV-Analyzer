import numpy as np
import pandas as pd
MISSING_WARNING_THRESHOLD = 0.05
MISSING_CRITICAL_THRESHOLD = 0.30
HIGH_CARDINALITY_THRESHOLD = 0.50
SKEWNESS_THRESHOLD = 2.0

def _get_numeric_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=np.number).columns.tolist()
    return [column for column, info in variable_types.items() if (column in df.columns and info.get("detected_type") == "numeric")]

def _get_categorical_columns(df, variable_types=None):
    if variable_types is None:
        return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_types = {"categorical", "categorical_numeric", "binary", "boolean"}
    return [column for column, info in variable_types.items() if (column in df.columns and info.get("detected_type") in categorical_types)]

def check_missing_values(df):
    results = {}
    total_rows = len(df)
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_ratio = (missing_count / total_rows if total_rows > 0 else 0)
        if (missing_ratio >= MISSING_CRITICAL_THRESHOLD):
            level = "critical"
        elif (missing_ratio >= MISSING_WARNING_THRESHOLD):
            level = "warning"
        else:
            level = "ok"
        results[column] = {"count": missing_count, "ratio": float(missing_ratio), "level": level}
    total_missing = int(df.isna().sum().sum())
    return {"total_missing": total_missing, "columns": results}

def check_duplicates(df):
    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)
    duplicate_ratio = (duplicate_count / total_rows if total_rows > 0 else 0)
    return {"count": duplicate_count, "ratio": float(duplicate_ratio)}

def check_constant_columns(df):
    constant_columns = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)
        if unique_count <= 1:
            constant_columns.append(column)
    return {"count": len(constant_columns), "columns": constant_columns}

def check_high_cardinality(df, variable_types=None):
    results = {}
    categorical_columns = (_get_categorical_columns(df, variable_types))
    total_rows = len(df)
    for column in categorical_columns:
        unique_count = int(df[column].nunique(dropna=True))
        ratio = (unique_count / total_rows if total_rows > 0 else 0)
        level = ("high" if ratio >= HIGH_CARDINALITY_THRESHOLD else "normal")
        results[column] = {"unique": unique_count, "ratio": float(ratio), "level": level}
    return results

def check_outliers(df, variable_types=None):
    results = {}
    numeric_columns = _get_numeric_columns(df, variable_types)
    for column in numeric_columns:
        series = (df[column].replace([np.inf, -np.inf], np.nan).dropna())
        if len(series) == 0:
            continue
        if series.nunique() <= 1:
            results[column] = {
                "count": 0,
                "ratio": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "status": "skipped",
                "reason": "constant_variable"
            }
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = (q1 - 1.5 * iqr)
        upper_bound = (q3 + 1.5 * iqr)
        mask = ((series < lower_bound) | (series > upper_bound))
        outlier_count = int(mask.sum())
        ratio = (outlier_count / len(series))
        results[column] = {
            "count": outlier_count,
            "ratio": float(ratio),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "status": "ok",
            "reason": None
        }
    return results

def check_skewness(df, variable_types=None):
    results = {}
    numeric_columns = _get_numeric_columns(df, variable_types)
    for column in numeric_columns:
        series = (df[column].replace([np.inf, -np.inf], np.nan).dropna())
        if len(series) < 3:
            results[column] = {"skewness": None, "level": "skipped", "status": "skipped", "reason": "insufficient_samples"}
            continue
        if series.nunique() <= 1:
            results[column] = {"skewness": 0.0, "level": "constant", "status": "skipped", "reason": "constant_variable"}
            continue
        skewness = float(series.skew())
        absolute_skewness = abs(skewness)
        if (absolute_skewness >= SKEWNESS_THRESHOLD):
            level = "strongly_skewed"
        elif absolute_skewness >= 1:
            level = "moderately_skewed"
        else:
            level = "approximately_symmetric"
        results[column] = {"skewness": skewness, "level": level, "status": "ok", "reason": None}
    return results

def check_type_issues(df, variable_types=None):
    results = {}
    for column in df.columns:
        dtype = str(df[column].dtype)
        detected_type = None
        issue = None
        if variable_types is not None:
            detected_type = (variable_types.get(column, {}).get("detected_type"))
        if detected_type == "text":
            issue = "text_variable"
        elif detected_type == "identifier":
            issue = "identifier_variable"
        elif detected_type in {"datetime", "time"}:
            issue = "temporal_variable"
        elif dtype == "object":
            issue = "object_dtype_requires_review"
        results[column] = {"dtype": dtype, "detected_type": detected_type, "issue": issue}
    return results

def analyze_quality(df, variable_types=None):
    missing = check_missing_values(df)
    duplicates = check_duplicates(df)
    constant_columns = (check_constant_columns(df))
    high_cardinality = (check_high_cardinality(df, variable_types))
    outliers = check_outliers(df, variable_types)
    skewness = check_skewness(df, variable_types)
    type_issues = check_type_issues(df, variable_types)
    return {
        "summary": {"rows": len(df), "columns": len(df.columns)},
        "missing": missing,
        "duplicates": duplicates,
        "constant_columns": constant_columns,
        "high_cardinality": high_cardinality,
        "outliers": outliers,
        "skewness": skewness,
        "type_issues": type_issues
    }
