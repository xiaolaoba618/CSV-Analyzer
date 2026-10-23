import sys
import csv
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtCore import (Qt, Signal, QObject, QThread, QTimer,)
from PySide6.QtGui import (QColor, QCursor, QPainter, QPen,)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QProgressBar,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QHeaderView,
    QDialogButtonBox,
    QToolTip,
)
from analyzer import analyze_csv
from report.dashboard_generator import (
    FULL_EXPORT_ROW_LIMIT,
    MAX_IDENTIFIER_VALUES,
    format_size,
)
from tools.data_loader import load_csv_preview, SUPPORTED_ENCODINGS
from tools.data_profiler import analyze_profile, apply_variable_type_overrides

def count_csv_rows(path):
    """快速统计 CSV 的数据行数（不含表头），用于体积预估。"""
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                try:
                    next(reader)
                except StopIteration:
                    return 0
                return sum(1 for _ in reader)
        except (UnicodeDecodeError, LookupError):
            continue
    return 0

EXPORT_ALL_TIP = (
    "<b>导出全部数据 —— 影响范围</b><br><br>"
    "报告里的数据分四类，本选项只影响第一类：<br><br>"
    "<b>1. 行级明细（受本选项控制，默认勾选）</b><br>"
    "• 勾选（默认）：交互式报告写入全部数据行，可逐行查看、按标识符查完整记录<br>"
    f"• 超过 {FULL_EXPORT_ROW_LIMIT:,} 行时会自动取消勾选，改为代表性抽样 "
    f"{FULL_EXPORT_ROW_LIMIT:,} 行（按时间等距分层，并按分类变量比例分配名额，"
    "尽量保留整体趋势与分布）；确认可接受体积可手动勾选回全量<br>"
    "• 静态报告（report.html）的统计图<b>始终使用全量数据</b>，与本选项无关<br><br>"
    "<b>2. 时间趋势（随行级数据一起变化）</b><br>"
    "趋势图不再单独存储聚合序列，而是由报告内的行级数据<b>实时按时间桶聚合</b>，"
    "因此口径与本选项一致：全量时是全部原始行，抽样时是这 "
    f"{FULL_EXPORT_ROW_LIMIT:,} 行。<br><br>"
    "<b>3. 标识符：按单个／组合标识符筛选</b><br>"
    "若某个变量被指定为「标识符」（如股票代码、用户 ID、设备号），可在报告里按"
    "单个标识符或多个标识符的组合筛选图表。<br>"
    "筛选读的是<b>同一份行级数据</b>，因此同样受本选项控制：勾选后可查到该对象的全部记录；"
    "未勾选时，样本外的标识符会提示「未包含在样本中」。<br>"
    "用于搜索补全的标识符取值清单恒为全量，不受影响。<br><br>"
    "<b>4. 恒为全量</b><br>"
    "KPI、缺失率、相关系数等统计量始终基于完整数据计算，与勾选无关。<br><br>"
    "<b>关于体积</b><br>"
    "体积主要取决于写入的行数与列构成，界面下方会给出生成前的估算值（误差约 15%）；"
    "分析完成后会显示两份报告的实际占用。<br>"
    "经验参考：约 50 MB 以内浏览器打开流畅，100 MB 以上会明显变慢。"
)

EXPORT_ALL_ADVICE = (
    f"默认勾选：行级明细不再重复存储，全量导出的开销已大幅下降。\n"
    f"超过 {FULL_EXPORT_ROW_LIMIT:,} 行时会自动改为抽样，需要全量请手动勾选；"
    "行数越多，报告体积越大、浏览器打开越慢。"
)

def estimate_dashboard_mb(table_rows, source_rows, variable_types):
    """按列构成粗略估算交互式报告体积（MB），仅用于界面提示。

    table_rows 为写入报告的行数（全量或抽样），source_rows 为源数据行数
    （标识符取值字典恒按全量计算，不受抽样影响）。
    """
    if not source_rows:
        return 0.0
    types = variable_types or {}
    numeric = sum(1 for info in types.values() if info.get("detected_type") == "numeric")
    identifier = sum(1 for info in types.values() if info.get("detected_type") == "identifier")
    other = max(len(types) - numeric - identifier, 0)
    per_row = numeric * 9 + identifier * 7 + other * 8
    mb = 1024.0 * 1024.0
    # 表内标识符取值字典随写入行数增长（列式存储每个取值只存一次）
    table_dict_mb = identifier * min(table_rows, source_rows) * 14 / mb
    # 标识符搜索清单与组合计数：不受抽样影响，但有取值上限
    search_dict_mb = identifier * min(source_rows, MAX_IDENTIFIER_VALUES) * 18 / mb
    combination_mb = (
        min(source_rows, MAX_IDENTIFIER_VALUES) * 18 / mb if identifier >= 2 else 0.0
    )
    return table_rows * per_row / mb + table_dict_mb + search_dict_mb + combination_mb + 3.0


class InfoBadge(QLabel):
    """圆圈感叹号提示图标：鼠标悬停满 0.5 秒后弹出说明。"""

    HOVER_DELAY_MS = 500

    def __init__(self, tooltip_text="", parent=None):
        super().__init__(parent)
        self.tooltip_text = tooltip_text
        self.setFixedSize(16, 16)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        # 置空可屏蔽 Qt 自带的即时 tooltip，改由下面自己控制延迟。
        self.setToolTip("")
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(self.HOVER_DELAY_MS)
        self.hover_timer.timeout.connect(self.show_tooltip)

    def set_tip(self, text):
        """更新悬停时显示的说明（支持富文本）。"""
        self.tooltip_text = text or ""
        QToolTip.hideText()

    def show_tooltip(self):
        if not self.tooltip_text:
            return
        QToolTip.showText(QCursor.pos(), self.tooltip_text, self)

    def enterEvent(self, event):
        self.hover_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.hover_timer.stop()
        self.show_tooltip()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#9aa1ab"), 1.2))
        painter.setBrush(QColor("#eef1f4"))
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        # 感叹号用几何图形绘制，避免字体缺字形时显示为方框。
        width, height = self.width(), self.height()
        center_x = width // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#5f6673"))
        bar_width = max(2, round(width * 0.16))
        bar_height = max(3, round(height * 0.34))
        bar_top = round(height * 0.20)
        painter.drawRoundedRect(center_x - bar_width // 2, bar_top, bar_width, bar_height, 1, 1)
        dot_size = max(2, round(width * 0.18))
        dot_top = round(height * 0.64)
        painter.drawEllipse(center_x - dot_size // 2, dot_top, dot_size, dot_size)
        painter.end()


class DropArea(QFrame):
    """CSV 文件拖放区域。"""
    file_dropped = Signal(Path)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if (path.is_file() and path.suffix.lower() == ".csv"):
                event.acceptProposedAction()
                return
        event.ignore()
    def dragMoveEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if (path.is_file() and path.suffix.lower() == ".csv"):
                event.acceptProposedAction()
                return
        event.ignore()
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if (path.is_file() and path.suffix.lower() == ".csv"):
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()

class AnalysisWorker(QObject):
    """在后台线程执行分析任务。"""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str, int)
    def __init__(self, input_file, output_dir, variable_type_overrides=None, export_all_data=False):
        super().__init__()
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.variable_type_overrides = (variable_type_overrides or {})
        self.export_all_data = bool(export_all_data)
    def report_progress(self, message, percentage):
        self.progress.emit(message, percentage)
    def run(self):
        try:
            result = analyze_csv(
                input_file=self.input_file,
                output_dir=self.output_dir,
                progress_callback=self.report_progress,
                variable_type_overrides=(self.variable_type_overrides),
                export_all_data=self.export_all_data
            )
            self.finished.emit(result)
        except Exception as error:
            self.error.emit(str(error))

class VariableTypeDialog(QDialog):
    """变量类型设置对话框。"""
    OPTIONS = [("自动识别", None), ("数值变量", "numeric"), ("分类变量", "categorical"), ("标识符", "identifier"), ("时间变量", "temporal"),]
    def __init__(self, profile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("确认变量类型")
        self.setMinimumSize(820, 560)
        self.profile = profile
        self.overrides = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        title = QLabel("请确认变量类型")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #202124;"
        )
        layout.addWidget(title)
        warning = QLabel(
            "自动识别可能出现误判。你可以逐列指定“数值变量”“分类变量”“标识符”或“时间变量”；"
            "选择“自动识别”则沿用程序原有判断。标识符适用于玩家 UID、用户 ID、设备 ID 等实体键；"
            "时间变量适用于日期、时间、周数等有先后顺序的指标。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "color: #8a4b08; background: #fff7e6; "
            "border: 1px solid #f3d19c; border-radius: 8px; padding: 10px;"
        )
        layout.addWidget(warning)
        self.table = QTableWidget(len(profile.get("variable_types", {})), 3, self)
        self.table.setHorizontalHeaderLabels(["变量名称", "自动识别结果", "本次分析使用"])
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.combos = {}
        for row, (column, info) in enumerate(profile.get("variable_types", {}).items()):
            name_item = QTableWidgetItem(str(column))
            name_item.setToolTip(str(column))
            self.table.setItem(row, 0, name_item)
            auto_type = info.get("detected_type", "unknown")
            auto_item = QTableWidgetItem(str(auto_type))
            self.table.setItem(row, 1, auto_item)
            combo = QComboBox()
            for label, value in self.OPTIONS:
                combo.addItem(label, value)
            self.table.setCellWidget(row, 2, combo)
            self.combos[str(column)] = combo
        layout.addWidget(self.table, 1)
        note = QLabel("提示：只有明确选择类型的列会覆盖自动判断；" "其他列继续使用原有的日期、时间、文本、二元等自动语义识别。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280; font-size: 13px;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认并继续")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("全部自动识别")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_overrides(self):
        overrides = {}
        for column, combo in self.combos.items():
            value = combo.currentData()
            if value in {"numeric", "categorical", "identifier", "temporal"}:
                overrides[column] = value
        return overrides

class CSVAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.input_row_count = 0
        self.preview_frame = None
        self.preview_variable_types = {}
        self.preview_identifier_columns = []
        self.variable_type_overrides = {}
        self.output_dir = (PROJECT_ROOT / "output")
        self.analysis_thread = None
        self.analysis_worker = None
        self.setWindowTitle("CSV Analyzer")
        self.setMinimumSize(760, 560)
        self.setup_ui()
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 35, 40, 35)
        main_layout.setSpacing(20)
        title = QLabel("CSV Analyzer")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        subtitle = QLabel(
            "自动分析csv数据基本信息\n"
            "自动生成交互式报告"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        self.drop_area = DropArea()
        self.drop_area.setObjectName("dropArea")
        self.drop_area.file_dropped.connect(self.set_input_file)
        drop_layout = QVBoxLayout()
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_icon = QLabel("CSV")
        drop_icon.setObjectName("dropIcon")
        drop_icon.setAlignment(Qt.AlignCenter)
        drop_icon.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        drop_layout.addWidget(drop_icon)
        self.file_label = QLabel("将 CSV 文件拖到这里")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        drop_layout.addWidget(self.file_label)
        hint = QLabel("或者点击下方按钮选择文件")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        drop_layout.addWidget(hint)
        self.drop_area.setLayout(drop_layout)
        main_layout.addWidget(self.drop_area)
        self.select_button = QPushButton("选择 CSV 文件")
        self.select_button.clicked.connect(self.select_file)
        main_layout.addWidget(self.select_button)
        self.selected_file_label = QLabel("尚未选择文件")
        self.selected_file_label.setObjectName("selectedFile")
        self.selected_file_label.setAlignment(Qt.AlignCenter)
        self.selected_file_label.setWordWrap(True)
        main_layout.addWidget(self.selected_file_label)
        output_row = QWidget()
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.setSpacing(8)
        output_caption = QLabel("报告导出目录")
        output_caption.setObjectName("outputCaption")
        output_row_layout.addWidget(output_caption)
        self.output_dir_edit = QLineEdit(str(self.output_dir))
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setObjectName("outputDirEdit")
        self.output_dir_edit.setToolTip(str(self.output_dir))
        output_row_layout.addWidget(self.output_dir_edit, 1)
        self.output_dir_button = QPushButton("选择目录")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_row_layout.addWidget(self.output_dir_button)
        main_layout.addWidget(output_row)
        export_row = QWidget()
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        export_row_layout.setSpacing(6)
        self.export_all_checkbox = QCheckBox("导出全部数据到交互式报告")
        self.export_all_checkbox.setObjectName("exportAllCheckbox")
        self.export_all_checkbox.setChecked(True)
        self.export_all_checkbox.setEnabled(False)
        self.export_all_checkbox.stateChanged.connect(self.update_export_hint)
        export_row_layout.addWidget(self.export_all_checkbox)
        self.export_info_badge = InfoBadge(EXPORT_ALL_TIP)
        self.export_info_badge.setObjectName("exportInfoBadge")
        export_row_layout.addWidget(self.export_info_badge)
        export_row_layout.addStretch(1)
        main_layout.addWidget(export_row)
        self.export_advice_label = QLabel(EXPORT_ALL_ADVICE)
        self.export_advice_label.setObjectName("exportAdvice")
        self.export_advice_label.setWordWrap(True)
        self.export_advice_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.export_advice_label)
        self.export_hint_label = QLabel("尚未选择文件")
        self.export_hint_label.setObjectName("exportHint")
        self.export_hint_label.setWordWrap(True)
        self.export_hint_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.export_hint_label)
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.setObjectName("analyzeButton")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.start_analysis)
        main_layout.addWidget(self.analyze_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("等待选择 CSV 文件")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        result_layout = QHBoxLayout()
        self.dashboard_button = QPushButton("打开交互式分析报告")
        self.dashboard_button.setEnabled(False)
        self.dashboard_button.clicked.connect(self.open_dashboard)
        self.report_button = QPushButton("打开静态分析报告")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.open_report)
        result_layout.addWidget(self.dashboard_button)
        result_layout.addWidget(self.report_button)
        main_layout.addLayout(result_layout)
        self.setLayout(main_layout)
        self.setStyleSheet(
            """
            QWidget {
                background: #f7f8fa;
                color: #202124;
                font-family: "Microsoft YaHei";
                font-size: 14px;
            }

            QLabel#title {
                font-size: 32px;
                font-weight: 700;
                color: #202124;
            }

            QLabel#subtitle {
                font-size: 15px;
                color: #6b7280;
            }

            QFrame#dropArea {
                background: white;
                border: 2px dashed #b8bec8;
                border-radius: 18px;
                min-height: 190px;
            }

            QFrame#dropArea:hover {
                border-color: #4f46e5;
            }

            QLabel#dropIcon {
                font-size: 30px;
                font-weight: 700;
                color: #4f46e5;
            }

            QLabel#fileLabel {
                font-size: 18px;
                font-weight: 600;
            }

            QLabel#hint {
                color: #8a9099;
            }

            QLabel#selectedFile {
                color: #4f46e5;
                font-size: 13px;
            }

            QLabel#outputCaption {
                color: #555b66;
                font-size: 13px;
            }

            QLineEdit#outputDirEdit {
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 7px 10px;
                color: #6b7280;
                font-size: 12px;
            }

            QLabel#exportAdvice {
                font-size: 12px;
                color: #6b7280;
                padding: 0 2px 2px 2px;
            }

            QCheckBox#exportAllCheckbox {
                font-size: 14px;
                padding: 2px 0;
            }

            QCheckBox#exportAllCheckbox::indicator {
                width: 16px;
                height: 16px;
            }

            QLabel#exportHint {
                font-size: 13px;
                padding: 10px 12px;
                border-radius: 8px;
                background: #f1f3f5;
                border: 1px solid #e3e6ea;
                color: #555b66;
            }

            QPushButton {
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 11px 18px;
                font-size: 14px;
            }

            QPushButton:hover {
                background: #f1f3f5;
            }

            QPushButton:disabled {
                color: #9ca3af;
                background: #eeeeee;
            }

            QPushButton#analyzeButton {
                background: #4f46e5;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: 600;
                padding: 13px;
            }

            QPushButton#analyzeButton:hover {
                background: #4338ca;
            }

            QLabel#status {
                color: #555b66;
                min-height: 30px;
            }

            QProgressBar#progressBar {
                border: none;
                border-radius: 5px;
                background: #e5e7eb;
                min-height: 8px;
                max-height: 8px;
                text-align: center;
            }

            QProgressBar#progressBar::chunk {
                background: #4f46e5;
                border-radius: 5px;
            }
            """
        )
    def select_file(self):
        if self.analysis_thread is not None:
            return
        file_path, _ = (QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV Files (*.csv)"))
        if not file_path:
            return
        self.set_input_file(Path(file_path))
    def select_output_dir(self):
        """选择报告导出目录，默认为项目根目录下的 output。"""
        start_dir = str(self.output_dir)
        if not Path(start_dir).is_dir():
            start_dir = str(PROJECT_ROOT)
        chosen = QFileDialog.getExistingDirectory(
            self,
            "选择报告导出目录",
            start_dir
        )
        if not chosen:
            return
        self.set_output_dir(Path(chosen))
    def set_output_dir(self, path):
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            QMessageBox.warning(self, "目录不可用", f"无法使用该目录：\n{path}\n{error}")
            return
        self.output_dir = path
        self.output_dir_edit.setText(str(path))
        self.output_dir_edit.setToolTip(str(path))
        self.output_dir_edit.setCursorPosition(0)
    def set_input_file(self, path):
        if self.analysis_thread is not None:
            return
        path = Path(path)
        if not path.exists():
            return
        if not path.is_file():
            return
        if path.suffix.lower() != ".csv":
            QMessageBox.warning(self, "文件格式错误", "请选择 CSV 文件。")
            return
        try:
            preview_df = load_csv_preview(path)
            preview_profile = analyze_profile(preview_df)
        except Exception as error:
            QMessageBox.critical(self, "CSV 读取失败", str(error))
            return
        dialog = VariableTypeDialog(preview_profile, parent=self)
        dialog.exec()
        self.variable_type_overrides = dialog.get_overrides()
        self.input_file = path
        self.preview_frame = preview_df
        preview_profile = apply_variable_type_overrides(preview_profile, self.variable_type_overrides)
        self.preview_variable_types = preview_profile.get("variable_types", {})
        self.preview_identifier_columns = [
            str(column)
            for column, info in self.preview_variable_types.items()
            if info.get("detected_type") == "identifier"
        ]
        try:
            self.input_row_count = count_csv_rows(path)
        except Exception:
            self.input_row_count = int(len(preview_df))
        self.apply_export_threshold()
        self.selected_file_label.setText(
            f"已选择：{path}"
        )
        self.file_label.setText(path.name)
        self.analyze_button.setEnabled(True)
        self.dashboard_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.status_label.setText("文件已准备好，可以开始分析。")
        self.update_export_hint()
    def apply_export_threshold(self):
        """超过阈值时默认改为抽样，用户仍可手动勾选回全量。"""
        if int(self.input_row_count or 0) <= FULL_EXPORT_ROW_LIMIT:
            return
        self.export_all_checkbox.blockSignals(True)
        self.export_all_checkbox.setChecked(False)
        self.export_all_checkbox.blockSignals(False)
    def update_export_hint(self):
        """根据数据行数与勾选状态，更新“导出全部数据”下方的说明与体积预估。"""
        neutral_style = (
            "font-size: 13px; padding: 10px 12px; border-radius: 8px; "
            "background: #f1f3f5; border: 1px solid #e3e6ea; color: #555b66;"
        )
        notice_style = (
            "font-size: 13px; padding: 10px 12px; border-radius: 8px; "
            "background: #fff7e6; border: 1px solid #f3d19c; color: #8a4b08;"
        )
        warning_style = (
            "font-size: 13px; padding: 10px 12px; border-radius: 8px; "
            "background: #fdecea; border: 1px solid #f5c6c2; color: #8a2b22;"
        )
        if self.input_file is None:
            self.export_all_checkbox.setEnabled(False)
            self.export_hint_label.setText("尚未选择文件")
            self.export_hint_label.setStyleSheet(neutral_style)
            return
        self.export_all_checkbox.setEnabled(True)
        rows = int(self.input_row_count or 0)
        identifier_note = ""
        if getattr(self, "preview_identifier_columns", None):
            identifier_note = (
                "<br><br><b>当前数据集</b><br>"
                f"共 {rows:,} 行；"
                f"标识符：{'、'.join(self.preview_identifier_columns[:3])}"
                f"{' 等' if len(self.preview_identifier_columns) > 3 else ''}，"
                "可用于报告中按单个取值筛选（受本选项控制）。"
            )
        else:
            identifier_note = f"<br><br><b>当前数据集</b><br>共 {rows:,} 行。"
        self.export_info_badge.set_tip(EXPORT_ALL_TIP + identifier_note)
        full_mb = estimate_dashboard_mb(rows, rows, self.preview_variable_types)
        sampled_mb = estimate_dashboard_mb(
            min(rows, FULL_EXPORT_ROW_LIMIT), rows, self.preview_variable_types
        )
        if rows <= FULL_EXPORT_ROW_LIMIT:
            self.export_hint_label.setStyleSheet(neutral_style)
            self.export_hint_label.setText(
                f"共 {rows:,} 行，未超过抽样上限 · 勾选与否报告一致（约 {full_mb:.0f} MB）"
            )
            return
        if self.export_all_checkbox.isChecked():
            if rows > FULL_EXPORT_ROW_LIMIT:
                self.export_hint_label.setStyleSheet(warning_style)
                self.export_hint_label.setText(
                    f"已手动勾选：全部 {rows:,} 行写入报告，约 {full_mb:.0f} MB · "
                    "浏览器打开会明显变慢"
                )
            else:
                self.export_hint_label.setStyleSheet(neutral_style)
                self.export_hint_label.setText(
                    f"默认全量：{rows:,} 行全部写入报告，约 {full_mb:.0f} MB"
                )
            return
        self.export_hint_label.setStyleSheet(notice_style)
        self.export_hint_label.setText(
            f"共 {rows:,} 行，超过 {FULL_EXPORT_ROW_LIMIT:,} 行阈值，已改为抽样 "
            f"{FULL_EXPORT_ROW_LIMIT:,} 行（约 {sampled_mb:.0f} MB）· "
            f"需要全量请勾选（约 {full_mb:.0f} MB）"
        )
    def start_analysis(self):
        if self.input_file is None:
            return
        if self.analysis_thread is not None:
            return
        if not self.output_dir.is_dir():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as error:
                QMessageBox.warning(self, "目录不可用", f"无法创建导出目录：\n{self.output_dir}\n{error}")
                self.analyze_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.status_label.setText("导出目录不可用，请重新选择。")
                return
        self.output_dir_edit.setEnabled(False)
        self.output_dir_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.export_all_checkbox.setEnabled(False)
        self.dashboard_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.status_label.setText("正在准备分析...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(
            input_file=self.input_file,
            output_dir=self.output_dir,
            variable_type_overrides=self.variable_type_overrides,
            export_all_data=self.export_all_checkbox.isChecked()
        )
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.finished.connect(self.analysis_finished)
        self.analysis_worker.error.connect(self.analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.error.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_thread_finished)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        self.analysis_thread.start()
    def update_progress(self, message, percentage):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    def analysis_finished(self, result):
        try:
            self.progress_bar.setValue(100)
            self.dashboard_path = Path(result["dashboard_path"])
            self.report_path = Path(result["report_path"])
            self.dashboard_button.setEnabled(True)
            self.report_button.setEnabled(True)
            # 生成前无法准确预估体积，这里在完成后直接给出实际占用。
            dashboard_size = self.measure_file_size(self.dashboard_path)
            report_size = self.measure_file_size(self.report_path)
            self.status_label.setText(
                f"分析完成！交互式报告 {dashboard_size} · 静态报告 {report_size}"
            )
        except Exception as error:
            self.analysis_failed(str(error))
    @staticmethod
    def measure_file_size(path):
        """读取文件实际占用大小，返回可读文本；读取失败时返回「未知」。"""
        try:
            return format_size(Path(path).stat().st_size)
        except OSError:
            return "未知"
    def analysis_failed(self, error_message):
        self.status_label.setText("分析失败。")
        QMessageBox.critical(self, "分析失败", error_message)
    def analysis_thread_finished(self):
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(self.input_file is not None)
        self.output_dir_edit.setEnabled(True)
        self.output_dir_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.export_all_checkbox.setEnabled(self.input_file is not None)
        self.analysis_worker = None
        self.analysis_thread = None
    def open_dashboard(self):
        if not hasattr(self, "dashboard_path"):
            return
        self.open_file(self.dashboard_path)
    def open_report(self):
        if not hasattr(self, "report_path"):
            return
        self.open_file(self.report_path)
    @staticmethod
    def open_file(path):
        path = Path(path)
        if not path.exists():
            QMessageBox.warning(
                None,
                "文件不存在",
                f"找不到文件：\n{path}"
            )
            return
        if sys.platform.startswith("win"):
            import os
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    def closeEvent(self, event):
        if self.analysis_thread is not None:
            QMessageBox.warning(self, "正在分析", "CSV 正在分析，请等待分析完成后再关闭程序。")
            event.ignore()
            return
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = CSVAnalyzerGUI()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
