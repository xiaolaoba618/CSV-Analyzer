#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CSV Analyzer 打包脚本。

用法：
    python build_exe.py                # 打包成单文件 exe（无控制台窗口）
    python build_exe.py --console      # 保留控制台窗口，便于排查启动报错
    python build_exe.py --onedir       # 打包成文件夹（更稳定，报告同样在 exe 同级）
    python build_exe.py --no-portable  # 报告改放到系统临时目录（exe 所在目录不可写时用）

打包完成后，产物位于：
    dist/CSV-Analyzer.exe

关于「报告输出目录」：
    gui/app.py 中通过 PROJECT_ROOT = Path(__file__).resolve().parent.parent 计算输出目录。
    在 PyInstaller 单文件模式下，入口脚本的 __file__ 会指向运行时的临时解压目录
    （%TEMP%\\_MEIxxxxxx\\app.py），因此父目录会变成系统临时目录，
    报告就会落到 %TEMP%\\output 而不是 exe 旁边。

    本脚本通过 --runtime-tmpdir . 让依赖解压到「启动时的工作目录」下，
    这样 __file__ 的父目录恰好等于 exe 所在目录，报告便会输出到 exe 同级的 output 目录。
    解压出来的 _MEIxxxxxx 目录会在程序退出时自动删除，不会残留。

    注意事项：
    * 请直接双击 exe 运行（双击时工作目录就是 exe 所在目录）。
      如果从其他目录用命令行启动，报告会输出到那个目录。
    * exe 所在目录必须可写（不要放在 C:\\Program Files 等只读位置）。
      若放在 OneDrive 等同步目录，运行时解压可能触发大量文件同步，
      此时建议改用 `--no-portable`，报告将输出到系统临时目录。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 基本配置

PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_ROOT / "gui" / "app.py"
APP_NAME = "CSV-Analyzer"

# 打包时必须一并带入的数据文件：(源文件, 打包后的相对目录)
# report/generator.py 会读取 Path(__file__).parent / "template.html"，
# analyzer.py 会读取 PROJECT_ROOT / "report" / "dashboard_template.html"，
# 因此两个模板都必须放进打包后的 report/ 目录，否则运行时会报模板找不到。
DATA_FILES = [
    ("report/template.html", "report"),
    ("report/dashboard_template.html", "report"),
]

# 运行时依赖：模块名 -> 安装提示
REQUIRED_MODULES = {
    "PySide6": "PySide6>=6.7,<7",
    "pandas": "pandas>=2.2,<3",
    "numpy": "numpy>=2.0,<3",
    "scipy": "scipy>=1.14,<2",
    "matplotlib": "matplotlib>=3.8,<4",
}

# 明确排除与本应用无关的模块，减小体积（这些模块本项目不会用到）
# 注意：不要排除 unittest —— numpy.testing / scipy 等在部分路径下会引用它。
EXCLUDED_MODULES = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "test",
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]


def log(message: str) -> None:
    print(f"[build_exe] {message}", flush=True)


# ---------------------------------------------------------------- 环境检查

def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"打包需要 Python 3.10 及以上版本，当前为 {sys.version.split()[0]}。"
        )


def check_source_files() -> None:
    if not ENTRY_SCRIPT.is_file():
        raise SystemExit(f"找不到 GUI 入口文件：{ENTRY_SCRIPT}")
    for source, _ in DATA_FILES:
        path = PROJECT_ROOT / source
        if not path.is_file():
            raise SystemExit(f"找不到需要打包的数据文件：{path}")


def check_runtime_dependencies() -> None:
    missing = [
        hint
        for module, hint in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    lines = "\n".join(f"    {name}" for name in missing)
    raise SystemExit(
        "当前 Python 环境缺少以下运行依赖，无法打包：\n"
        f"{lines}\n\n"
        "请先安装依赖后重试：\n"
        "    pip install -r requirements.txt"
    )


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    log("未检测到 PyInstaller，正在自动安装……")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pyinstaller"]
    )
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("PyInstaller 安装失败，请手动执行：pip install pyinstaller")


# ---------------------------------------------------------------- 打包

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PyInstaller 将 CSV Analyzer 打包为可执行文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=APP_NAME,
        help=f"生成的 exe 名称（默认：{APP_NAME}）",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="保留控制台窗口，便于查看启动/运行报错（默认不保留）。",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="打包成文件夹而不是单个 exe（体积更小、启动更快、更稳定）。",
    )
    parser.add_argument(
        "--no-portable",
        action="store_true",
        help=(
            "不指定 --runtime-tmpdir，报告将输出到系统临时目录（%TEMP%%\\output）。"
            "当 exe 所在目录不可写时使用。"
        ),
    )
    parser.add_argument(
        "--icon",
        default=None,
        help="可选：exe 图标文件路径（.ico）。",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="保留中间构建产物（build 目录）。",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """组装 PyInstaller 命令行参数。"""
    work_path = PROJECT_ROOT / "build" / "pyinstaller"
    dist_path = PROJECT_ROOT / "dist"
    spec_path = PROJECT_ROOT / "build"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        # 让打包后的程序能找到项目内的顶层模块（analyzer / tools / report / gui）
        "--paths",
        str(PROJECT_ROOT),
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
    ]

    command.append("--onedir" if args.onedir else "--onefile")
    # GUI 程序默认不显示控制台窗口
    command.append("--console" if args.console else "--windowed")

    if args.icon:
        icon_path = Path(args.icon).expanduser().resolve()
        if not icon_path.is_file():
            raise SystemExit(f"找不到图标文件：{icon_path}")
        command += ["--icon", str(icon_path)]

    if not args.no_portable:
        # 关键：让依赖解压目录落在「启动时的工作目录」。
        # 双击运行时工作目录 = exe 所在目录，于是 gui/app.py 里的
        # PROJECT_ROOT 会等于 exe 目录，报告输出到 exe 同级的 output 目录。
        command += ["--runtime-tmpdir", "."]

    # 两个 HTML 模板必须随包分发到 report/ 目录
    for source, destination in DATA_FILES:
        command += [
            "--add-data",
            f"{PROJECT_ROOT / source}{os.pathsep}{destination}",
        ]

    for module in EXCLUDED_MODULES:
        command += ["--exclude-module", module]

    command.append(str(ENTRY_SCRIPT))
    return command


def run_build(args: argparse.Namespace) -> Path:
    build_command_list = build_command(args)
    log("开始打包，完整命令如下：")
    print("    " + " ".join(build_command_list))
    print()

    result = subprocess.run(build_command_list, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"打包失败，PyInstaller 退出码：{result.returncode}")

    if args.onedir:
        output = PROJECT_ROOT / "dist" / args.name / f"{args.name}.exe"
    else:
        output = PROJECT_ROOT / "dist" / f"{args.name}.exe"

    if not output.exists():
        raise SystemExit(f"打包结束但未找到产物：{output}")
    return output


def cleanup_build_directory(keep: bool) -> None:
    if keep:
        return
    build_directory = PROJECT_ROOT / "build"
    if build_directory.exists():
        shutil.rmtree(build_directory, ignore_errors=True)


def print_summary(output: Path, args: argparse.Namespace) -> None:
    size_mb = output.stat().st_size / 1024 / 1024
    print()
    print("=" * 64)
    print("打包完成")
    print("=" * 64)
    print(f"产物路径：{output}")
    print(f"文件大小：{size_mb:.1f} MB")
    print()
    print("使用方式：直接双击运行该 exe，无需安装 Python 或任何依赖。")
    if not args.onedir:
        print()
        print("说明：")
        if args.no_portable:
            print("  * 报告输出到系统临时目录：%TEMP%\\output")
            print("    （路径较隐蔽，适合 exe 放在只读目录时使用）")
        else:
            print("  * 报告输出到 exe 同级的 output 目录。")
            print("  * 请直接双击运行，程序会把依赖解压到当前目录并在退出时自动清理。")
            print("  * exe 所在目录需要可写；若放在 Program Files 等只读位置请改用 --no-portable。")
            print("  * 若放在 OneDrive 等同步目录，运行时解压可能触发大量同步，"
                  "建议改为 --no-portable 或放到本地非同步目录。")
    else:
        print()
        print("说明：")
        print("  * 分发时需要把整个 dist/{0} 文件夹一起打包给别人。".format(args.name))
        print("  * 报告输出到 exe 同级的 output 目录。")
    print()
    print("提示：如果双击后没有反应，请用 `python build_exe.py --console` 重新打包，")
    print("      在控制台中查看具体报错信息。")
    print("=" * 64)


def main() -> None:
    args = parse_arguments()
    check_python_version()
    check_source_files()
    check_runtime_dependencies()
    ensure_pyinstaller()

    if not sys.platform.startswith("win"):
        log("提示：当前系统不是 Windows，PyInstaller 无法交叉编译出 .exe。")

    output = run_build(args)
    cleanup_build_directory(args.keep_build)
    print_summary(output, args)


if __name__ == "__main__":
    main()
