# 自动化数据分析报告生成工具

**A Lightweight CSV Data Analyzer**

自动化数据分析报告生成工具 是一个基于 Python 开发的轻量级 CSV 数据自动分析工具，旨在帮助用户快速了解陌生数据集，简化初步数据探索流程，将更多精力投入后续的数据处理与建模。

本项目主要用于学习、研究和数据探索，欢迎交流与改进。

## 功能介绍

* **数据概览**：查看数据集的行数、列数、变量类型及基本结构。
* **统计摘要**：计算数值变量的描述性统计指标，检查缺失值和重复记录。
* **变量分析**：识别数值变量、分类变量等，并支持变量类型调整。
* **数据可视化**：自动生成变量分布、变量关系及时间变化等统计图表。
* **数据筛选**：支持按变量条件筛选数据，并查看筛选后的分析结果。
* **分析报告**：生成 HTML 统计报告和交互式 Dashboard，便于查看与分享。

## 风险与注意事项

* **分析结果仅供参考**：自动识别的变量类型和统计结果可能存在误差，建议结合数据背景进行核验。
* **不替代专业分析**：工具主要用于初步探索，不能替代严谨的数据清洗、统计推断和建模过程。
* **注意数据隐私**：请勿在不可信的环境中处理包含个人隐私或敏感信息的数据。
* **注意数据规模**：较大的 CSV 文件可能占用较多内存，影响运行速度。
* **报告依赖网络资源**：交互式 Dashboard 使用的部分前端资源可能需要联网加载。

## 使用教程

### 1. 环境要求

建议使用 Python 3.10 或更高版本。

### 2. 下载项目

```bash
git clone https://github.com/xiaolaoba618/csv-analyzer.git
cd csv-analyzer
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果项目尚未提供 `requirements.txt`，请根据实际使用的 Python 库安装依赖。

### 4. 启动程序

在项目根目录运行：

```bash
python main.py
```

如果项目入口文件名称不同，请替换为实际的 GUI 启动文件。

### 5. 分析数据

1. 启动程序并选择需要分析的 CSV 文件。
2. 根据需要调整变量类型和分析选项。
3. 执行分析并查看统计结果。
4. 在项目的 `output` 目录中查看生成的 HTML 报告和 Dashboard。

## 依赖要求

项目基于 Python 开发，主要依赖包括：

* Python
* PySide6：图形界面
* pandas：数据读取与处理
* NumPy：数值计算
* Matplotlib：统计图表
* Scipy：统计方法

具体依赖版本请以项目的 `requirements.txt` 为准。

## 部分图片展示

<img width="762" height="701" alt="f5314513b5a322c7c212194dc5eb568d" src="https://github.com/user-attachments/assets/c1bfa878-2b5d-4bf2-a95e-132ac3c94340" />


*图：工具GUI应用界面*


<img width="762" height="701" alt="f5314513b5a322c7c212194dc5eb568d" src="https://github.com/user-attachments/assets/59fe494a-a4ae-4cf4-8f5a-40d7acfe4fe1" />


*图：生成的交互式HTML报告/Dashboard界面*


## License

MIT
