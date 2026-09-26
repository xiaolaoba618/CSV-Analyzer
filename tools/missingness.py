import itertools
import numpy as np
import pandas as pd
DEFAULT_TOP_PATTERNS = 20
DEFAULT_TOP_PAIRS = 30

def _get_analysis_columns(df, variable_types=None):
    if variable_types is None:
        return df.columns.tolist()
    analysis_columns = []
    for column, info in variable_types.items():
        if column not in df.columns:
            continue
        detected_type = info.get("detected_type")
        if detected_type is not None:
            analysis_columns.append(column)
    return analysis_columns

def analyze_missingness_summary(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    if len(columns) == 0:
        return {
            "total_cells": 0,
            "total_missing": 0,
            "overall_missing_ratio": 0.0,
            "columns_with_missing": 0,
            "columns_without_missing": 0,
            "complete_rows": int(df.shape[0]),
            "complete_row_ratio": (1.0 if len(df) > 0 else 0.0),
            "rows_with_missing": 0,
            "rows_with_missing_ratio": (0.0)
        }
    subset = df[columns]
    total_cells = int(subset.shape[0] * subset.shape[1])
    total_missing = int(subset.isna().sum().sum())
    columns_with_missing = int(subset.isna().any().sum())
    columns_without_missing = int(subset.shape[1] - columns_with_missing)
    complete_rows = int(subset.notna().all(axis=1).sum())
    rows_with_missing = int(subset.isna().any(axis=1).sum())
    row_count = int(subset.shape[0])
    if total_cells > 0:
        overall_missing_ratio = (total_missing / total_cells)
    else:
        overall_missing_ratio = 0.0
    if row_count > 0:
        complete_row_ratio = (complete_rows / row_count)
        rows_with_missing_ratio = (rows_with_missing / row_count)
    else:
        complete_row_ratio = 0.0
        rows_with_missing_ratio = 0.0
    return {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "overall_missing_ratio": float(overall_missing_ratio),
        "columns_with_missing": (columns_with_missing),
        "columns_without_missing": (columns_without_missing),
        "complete_rows": complete_rows,
        "complete_row_ratio": float(complete_row_ratio),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_ratio": float(rows_with_missing_ratio)
    }

def analyze_variable_missingness(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    results = []
    row_count = int(df.shape[0])
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        missing_count = int(series.isna().sum())
        non_missing_count = int(series.notna().sum())
        if row_count > 0:
            missing_ratio = (missing_count / row_count)
        else:
            missing_ratio = 0.0
        results.append({
            "column": column,
            "missing_count": missing_count,
            "non_missing_count": (non_missing_count),
            "missing_ratio": float(missing_ratio)
        })
    results.sort(key=lambda item: (item["missing_ratio"], item["missing_count"]), reverse=True)
    return results

def analyze_missing_patterns(df, columns=None, top_n=DEFAULT_TOP_PATTERNS):
    if columns is None:
        columns = df.columns.tolist()
    if len(columns) == 0:
        return []
    subset = df[columns]
    missing_matrix = subset.isna()
    patterns = []
    for _, row in missing_matrix.iterrows():
        missing_columns = [column for column in columns if bool(row[column])]
        if not missing_columns:
            continue
        patterns.append(tuple(missing_columns))
    if not patterns:
        return []
    pattern_series = pd.Series(patterns, dtype="object")
    pattern_counts = (pattern_series.value_counts())
    row_count = int(len(df))
    results = []
    for pattern, count in pattern_counts.items():
        count = int(count)
        if row_count > 0:
            ratio = count / row_count
        else:
            ratio = 0.0
        missing_columns = list(pattern)
        results.append({
            "missing_columns": missing_columns,
            "missing_count": count,
            "missing_ratio": float(ratio),
            "missing_variable_count": len(missing_columns)
        })
        if len(results) >= top_n:
            break
    return results

def analyze_pairwise_missingness(df, columns=None, top_n=DEFAULT_TOP_PAIRS):
    if columns is None:
        columns = df.columns.tolist()
    if len(columns) < 2:
        return []
    subset = df[columns]
    results = []
    row_count = int(subset.shape[0])
    for column_1, column_2 in itertools.combinations(columns, 2):
        missing_1 = subset[column_1].isna()
        missing_2 = subset[column_2].isna()
        both_missing = (missing_1 & missing_2)
        both_missing_count = int(both_missing.sum())
        if row_count > 0:
            both_missing_ratio = (both_missing_count / row_count)
        else:
            both_missing_ratio = 0.0
        results.append({
            "column_1": column_1,
            "column_2": column_2,
            "both_missing_count": (both_missing_count),
            "both_missing_ratio": float(both_missing_ratio)
        })
    results.sort(key=lambda item: (item["both_missing_ratio"], item["both_missing_count"]), reverse=True)
    return results[:top_n]

def analyze_missingness(df, variable_types=None, top_patterns=DEFAULT_TOP_PATTERNS, top_pairs=DEFAULT_TOP_PAIRS):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    columns = _get_analysis_columns(df, variable_types)
    summary = analyze_missingness_summary(df, columns)
    variable_missingness = (analyze_variable_missingness(df, columns))
    missing_patterns = (analyze_missing_patterns(df, columns, top_n=top_patterns))
    pairwise_missingness = (analyze_pairwise_missingness(df, columns, top_n=top_pairs))
    return {
        "summary": summary,
        "variable_missingness": (variable_missingness),
        "missing_patterns": (missing_patterns),
        "pairwise_missingness": (pairwise_missingness),
        "columns": columns
    }
