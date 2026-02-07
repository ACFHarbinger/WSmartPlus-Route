from gui.src.components import ClickableHeaderWidget
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class TimeoutWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. Create a container widget for the header
        self.header_widget = ClickableHeaderWidget(self._toggle)
        self.header_widget.setStyleSheet("QWidget { border: none; padding: 0; margin-top: 5px; }")

        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        # 2. The main text
        self.label = QLabel("Timeout")
        self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button
        self.toggle_button = QPushButton("+")
        self.toggle_button.setFlat(True)
        self.toggle_button.setFixedSize(QSize(20, 20))
        self.toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.toggle_button.clicked.connect(self._toggle)

        # 4. Add components to header
        header_layout.addWidget(self.label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_button)

        layout.addRow(self.header_widget)

        # 6. Container for collapsible content
        self.content_container = QWidget()
        content_layout = QFormLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # --timeout input
        self.input = QLineEdit()
        self.input.setPlaceholderText("Timeout in seconds")
        content_layout.addRow(QLabel("Timeout (s):"), self.input)

        layout.addWidget(self.content_container)

        # Initialize state
        self.is_visible = False
        self.content_container.hide()

    def _toggle(self):
        if self.is_visible:
            self.content_container.hide()
            self.toggle_button.setText("+")
            self.label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.content_container.show()
            self.toggle_button.setText("-")
            self.label.setStyleSheet("QLabel { border: none; padding: 5px; background-color: transparent; }")
        self.is_visible = not self.is_visible

    def get_value(self):
        text = self.input.text().strip()
        if text:
            try:
                return int(text)
            except ValueError:
                print("Warning: timeout must be an integer. Defaulting to None.")
                return None
        return None
