import argparse
from pathlib import Path
from analyzer import analyze_csv

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=("CSV Analyzer - " "Automatic statistical analysis and " "interactive dashboard generation.")
    )
    parser.add_argument("input_file", type=Path, help="Path to the input CSV file.")
    parser.add_argument("--output", type=Path, default=Path("output"), help=("Output directory. " "Default: output"))
    return parser.parse_args()

def main():
    """启动 CSV Analyzer。"""
    args = parse_arguments()
    try:
        analyze_csv(input_file=args.input_file, output_dir=args.output)
    except FileNotFoundError as error:
        print(
            f"Error: {error}"
        )
    except ValueError as error:
        print(
            f"Error: {error}"
        )
    except Exception as error:
        print("Analysis failed:")
        print(error)
if __name__ == "__main__":
    main()
