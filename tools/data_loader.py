from pathlib import Path
import pandas as pd
SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk",]

def validate_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )
    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.suffix}"
        )
    return path

def read_raw_headers(file_path, encoding):
    try:
        header = pd.read_csv(file_path, encoding=encoding, header=None, nrows=1, dtype=str, keep_default_na=False)
        if header.empty:
            return []
        return header.iloc[0].tolist()
    except pd.errors.EmptyDataError:
        return []

def get_unnamed_column_positions(raw_headers):
    unnamed_positions = set()
    for index, header in enumerate(raw_headers):
        if header is None:
            unnamed_positions.add(index)
            continue
        header = str(header).strip()
        if not header:
            unnamed_positions.add(index)
    return unnamed_positions

def make_unique_columns(df):
    df = df.copy()
    seen = {}
    new_columns = []
    for column in df.columns:
        column = str(column).strip()
        if not column:
            column = "Unnamed"
        if column not in seen:
            seen[column] = 1
            new_columns.append(column)
        else:
            seen[column] += 1
            new_columns.append(
                f"{column}__{seen[column]}"
            )
    df.columns = new_columns
    return df

def remove_empty_columns(df, unnamed_positions=None):
    df = df.copy()
    if unnamed_positions is None:
        unnamed_positions = set()
    columns_to_remove = []
    for position in unnamed_positions:
        if position >= len(df.columns):
            continue
        column = df.columns[position]
        if df[column].isna().all():
            columns_to_remove.append(column)
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
    return df, columns_to_remove

def clean_column_names(df, unnamed_positions=None):
    df = df.copy()
    unnamed_positions = (unnamed_positions or set())
    cleaned_columns = []
    for position, column in enumerate(df.columns):
        column = str(column).strip()
        # 原始表头为空。
        if position in unnamed_positions:
            if not column or column.startswith("Unnamed:"):
                column = "Unnamed"
        # 原始表头只有空格。
        elif not column:
            column = "Unnamed"
        cleaned_columns.append(column)
    df.columns = cleaned_columns
    df = make_unique_columns(df)
    return df

def load_csv(file_path):
    path = validate_file(file_path)
    errors = []
    for encoding in SUPPORTED_ENCODINGS:
        try:
            # 读取完整数据集。
            df = pd.read_csv(path, encoding=encoding)
            # 检查空文件。
            if (df.empty and len(df.columns) == 0):
                raise ValueError("The CSV file is empty.")
            # 读取原始表头信息。
            raw_headers = (read_raw_headers(path, encoding))
            unnamed_positions = (get_unnamed_column_positions(raw_headers))
            df, _ = remove_empty_columns(df, unnamed_positions)
            # 规范化剩余列名。
            df = clean_column_names(df, unnamed_positions)
            return df
        except UnicodeDecodeError as error:
            errors.append(
                f"{encoding}: {error}"
            )
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError as error:
            raise ValueError(
                f"Unable to parse CSV file: {error}"
            )
    raise ValueError(
        "Unable to decode the CSV file. "
        f"Supported encodings: "
        f"{SUPPORTED_ENCODINGS}. "
        f"Details: {errors}"
    )

def get_loading_info(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": (df.columns.tolist()),
        "dtypes": {column: str(df[column].dtype) for column in df.columns},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

def load_csv_preview(file_path, nrows=1000):
    path = validate_file(file_path)
    errors = []
    for encoding in SUPPORTED_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, nrows=nrows,)
            if df.empty and len(df.columns) == 0:
                raise ValueError("The CSV file is empty.")
            raw_headers = read_raw_headers(path, encoding)
            unnamed_positions = get_unnamed_column_positions(raw_headers)
            # 预览阶段不因空值删除列，避免误删完整数据中的有效列。
            df = clean_column_names(df, unnamed_positions)
            return df
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError as error:
            raise ValueError(f"Unable to parse CSV file: {error}")
    raise ValueError(
        "Unable to decode the CSV file. "
        f"Supported encodings: {SUPPORTED_ENCODINGS}. "
        f"Details: {errors}"
    )

def load_and_prepare_csv(file_path):
    path = validate_file(file_path)
    df = load_csv(path)
    info = get_loading_info(df)
    info["file_path"] = str(path)
    return df, info
