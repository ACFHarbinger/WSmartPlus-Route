from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...styles.globals import START_GREEN_STYLE, START_RED_STYLE


class EvalDataBatchingTab(QWidget):
    """
    Tab for dataset slicing, batching, and system performance flags.
    """

    def __init__(self):
        super().__init__()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)

        form_layout.addRow(QLabel("<b>Data Processing</b>"))

        # --val_size
        self.val_size_input = QSpinBox(minimum=1, maximum=100_000, value=12_800)
        self.val_size_input.setSingleStep(100)
        form_layout.addRow("Validation Size:", self.val_size_input)

        # --offset
        self.offset_input = QSpinBox(minimum=0, maximum=100_000, value=0)
        form_layout.addRow("Dataset Offset:", self.offset_input)

        # --eval_batch_size
        self.eval_batch_size_input = QSpinBox(minimum=1, maximum=1024, value=256)
        form_layout.addRow("Evaluation Batch Size:", self.eval_batch_size_input)

        # --max_calc_batch_size
        self.max_calc_batch_size_input = QSpinBox(minimum=1, maximum=100_000, value=12_800)
        self.max_calc_batch_size_input.setSingleStep(100)
        form_layout.addRow("Maximum Calculation Sub-Batch Size:", self.max_calc_batch_size_input)

        form_layout.addRow(QLabel("<b>System Flags</b>"))

        # --no_cuda
        self.no_cuda_check = QPushButton("Use CUDA (Nvidia GPU)")
        self.no_cuda_check.setCheckable(True)
        self.no_cuda_check.setChecked(False)
        self.no_cuda_check.setStyleSheet(START_GREEN_STYLE)
        form_layout.addRow(QLabel("CUDA (disable for CPU only):"), self.no_cuda_check)

        # --no_progress_bar
        self.no_progress_bar_check = QPushButton("Progress Bar")
        self.no_progress_bar_check.setCheckable(True)
        self.no_progress_bar_check.setChecked(False)
        self.no_progress_bar_check.setStyleSheet(START_RED_STYLE)
        form_layout.addRow(QLabel("Progress Bar:"), self.no_progress_bar_check)

        # --compress_mask
        self.compress_mask_check = QPushButton("Compress Mask")
        self.compress_mask_check.setCheckable(True)
        self.compress_mask_check.setChecked(False)
        self.compress_mask_check.setStyleSheet(START_RED_STYLE)
        form_layout.addRow(QLabel("Compress Mask:"), self.compress_mask_check)

        # --multiprocessing
        self.multiprocessing_check = QPushButton("Use Multiprocessing")
        self.multiprocessing_check.setCheckable(True)
        self.multiprocessing_check.setChecked(False)
        self.multiprocessing_check.setStyleSheet(START_RED_STYLE)
        form_layout.addRow(QLabel("Multiprocessing:"), self.multiprocessing_check)

        scroll_area.setWidget(content)
        QVBoxLayout(self).addWidget(scroll_area)

    def get_params(self):
        params = {
            "val_size": self.val_size_input.value(),
            "offset": self.offset_input.value(),
            "eval_batch_size": self.eval_batch_size_input.value(),
            "max_calc_batch_size": self.max_calc_batch_size_input.value(),
            # Boolean flags
            "no_cuda": self.no_cuda_check.isChecked(),
            "no_progress_bar": self.no_progress_bar_check.isChecked(),
            "compress_mask": self.compress_mask_check.isChecked(),
            "multiprocessing": self.multiprocessing_check.isChecked(),
        }
        # Invert 'no_' flags for argparse logic if needed, but here we pass them as provided
        return {k: v for k, v in params.items() if not isinstance(v, bool) or v}
