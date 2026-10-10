import sys
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtCore import (Qt, Signal, QObject, QThread,)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QLabel,
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
)
from analyzer import analyze_csv
from tools.data_loader import load_csv_preview
from tools.data_profiler import analyze_profile

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
    def __init__(self, input_file, output_dir, variable_type_overrides=None):
        super().__init__()
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.variable_type_overrides = (variable_type_overrides or {})
    def report_progress(self, message, percentage):
        self.progress.emit(message, percentage)
    def run(self):
        try:
            result = analyze_csv(
                input_file=self.input_file,
                output_dir=self.output_dir,
                progress_callback=self.report_progress,
                variable_type_overrides=(self.variable_type_overrides)
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
        self.selected_file_label.setText(
            f"已选择：{path}"
        )
        self.file_label.setText(path.name)
        self.analyze_button.setEnabled(True)
        self.dashboard_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.status_label.setText("文件已准备好，可以开始分析。")
    def start_analysis(self):
        if self.input_file is None:
            return
        if self.analysis_thread is not None:
            return
        self.analyze_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.dashboard_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.status_label.setText("正在准备分析...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.output_dir = (PROJECT_ROOT / "output")
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(
            input_file=self.input_file,
            output_dir=self.output_dir,
            variable_type_overrides=self.variable_type_overrides
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
            self.status_label.setText("分析完成！")
            self.dashboard_path = Path(result["dashboard_path"])
            self.report_path = Path(result["report_path"])
            self.dashboard_button.setEnabled(True)
            self.report_button.setEnabled(True)
        except Exception as error:
            self.analysis_failed(str(error))
    def analysis_failed(self, error_message):
        self.status_label.setText("分析失败。")
        QMessageBox.critical(self, "分析失败", error_message)
    def analysis_thread_finished(self):
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(self.input_file is not None)
        self.select_button.setEnabled(True)
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
