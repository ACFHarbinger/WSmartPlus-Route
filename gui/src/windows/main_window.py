"""
Main application window for the WSmart-Route GUI.

This module defines the MainWindow class, which serves as the primary
interface for configuring and launching machine learning models and
operations research solvers.
"""

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.mediator import UIMediator
from ..styles.globals import (
    DARK_QSS,
    LIGHT_QSS,
    MUTED_TEXT_COLOR,
    TEXT_COLOR,
)
from .main import ProcessManager, TabManager


class MainWindow(QWidget):
    """
    The primary application window for the WSmart-Route framework.

    Manages the overall layout, command selection, theme toggling,
    and task-specific tabs. It also handles the execution of external
    CLI commands using QProcess.
    """

    def __init__(
        self,
        test_only=False,
        initial_window="Train Model",
        restart_callback=None,
        initial_tab_index=0,
    ):
        """
        Initialize the main application window.
        """
        super().__init__()
        self.test_only = test_only
        self.restart_callback = restart_callback
        self.setWindowTitle(
            "Machine Learning Models and Operations Research Solvers for Combinatorial Optimization Problems"
        )
        self.setMinimumSize(1080, 900)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialize Mediator
        self.mediator = UIMediator(self)
        self.mediator.command_updated.connect(self.update_preview_text)

        # Theme tracking
        self.current_theme = "light"
        self.setStyleSheet(LIGHT_QSS)

        # Components
        self.tab_manager = TabManager()
        self.process_manager = ProcessManager(self)
        # Expose maps for process manager access (kept for backward compatibility with process logic)
        self.test_sim_tabs_map = self.tab_manager.test_sim_tabs_map

        # Build UI
        self._init_ui(initial_window, initial_tab_index)

        # Register tabs
        self.tab_manager.register_tabs(self.mediator)

        # Connect signals
        self.process_manager.finished.connect(self.on_command_finished)

    def _init_ui(self, initial_window, initial_tab_index):
        """Build the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        self.title_label = QLabel("Machine Learning and Operations Research for Combinatorial Optimization")
        self.title_label.setObjectName("mainTitleLabel")
        main_layout.addWidget(self.title_label)

        # Command selection
        command_layout = QHBoxLayout()
        command_layout.addWidget(QLabel("Select Command:"))

        self.command_combo = QComboBox()
        self.command_combo.addItems(list(self.tab_manager.all_tabs.keys()))
        self.command_combo.currentTextChanged.connect(self.on_command_changed)
        command_layout.addWidget(self.command_combo)
        command_layout.addStretch()

        # Theme Toggle
        self.theme_toggle_button = QPushButton("🎨")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        self.theme_toggle_button.setFixedSize(25, 25)
        command_layout.addWidget(self.theme_toggle_button)

        main_layout.addLayout(command_layout)

        # Tabs container
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Preview
        preview_layout = QVBoxLayout()
        preview_label = QLabel("Generated Command (Read-Only):" if self.test_only else "Generated Command:")
        preview_label.setStyleSheet("font-weight: 600; padding-top: 5px;")
        preview_layout.addWidget(preview_label)

        self.preview = QTextEdit()
        self.preview.setObjectName("previewTextEdit")
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(180)
        preview_layout.addWidget(self.preview)

        # Restore logic
        self.command_combo.setCurrentText(initial_window)
        self.setup_tabs(initial_window)
        if initial_tab_index is not None:
            self.tabs.setCurrentIndex(initial_tab_index)

        self.update_preview()
        self.tabs.currentChanged.connect(self.update_preview)

        # Lower Controls
        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(15)
        lower_layout.addLayout(preview_layout, 3)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)

        self.reopen_button = QPushButton("Close and Reopen GUI")
        self.reopen_button.clicked.connect(self.close_and_reopen)
        control_layout.addWidget(self.reopen_button)

        secondary_style = f"background-color: #ECF0F1; color: {TEXT_COLOR};"

        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.update_preview)
        self.refresh_button.setStyleSheet(secondary_style)
        control_layout.addWidget(self.refresh_button)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setStyleSheet(secondary_style)
        control_layout.addWidget(self.copy_button)

        self.run_button = QPushButton("Run Command (simulated)" if self.test_only else "Run Command")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self.run_command)
        control_layout.addWidget(self.run_button)

        notes_label = QLabel(
            "Notes:\n• Leave fields empty to use defaults\n• Cost weights of 0 are ignored\n• Use Refresh to update preview"
        )
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet(f"font-size: 11px; color: {MUTED_TEXT_COLOR}; padding-top: 5px;")
        control_layout.addWidget(notes_label)

        lower_layout.addLayout(control_layout, 1)
        main_layout.addLayout(lower_layout)

    def toggle_theme(self):
        """Toggles the application stylesheet between light and dark mode."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.setStyleSheet(DARK_QSS if self.current_theme == "dark" else LIGHT_QSS)

    def close_and_reopen(self):
        """Hides the current window and triggers external restart."""
        current_tab_index = self.tabs.currentIndex()
        self.hide()
        if self.restart_callback:
            self.restart_callback(test_only=self.test_only, tab_index=current_tab_index)

    def setup_tabs(self, command):
        """Dynamically loads the correct set of tabs based on the command."""
        self.tab_manager.setup_tabs_in_widget(self.tabs, command)

    def on_command_changed(self, command):
        """Handle command selection change."""
        self.setup_tabs(command)
        self.update_preview()

    def update_preview_text(self, text):
        """Update preview text edit."""
        self.preview.setPlainText(text)

    def update_preview(self):
        """Delegate preview update to Mediator."""
        self.mediator.set_current_command(self.command_combo.currentText())

    def copy_to_clipboard(self):
        """Copy command to clipboard."""
        self.update_preview()
        QApplication.clipboard().setText(self.preview.toPlainText())
        QMessageBox.information(self, "Copied:", self.preview.toPlainText())

    def run_command(self):
        """Execute selected command."""
        self.run_button.setDisabled(True)
        self.process_manager.run_command(self.preview.toPlainText(), self.command_combo.currentText(), self.test_only)

    def on_command_finished(self, exit_code, exit_status):
        """Called when command finishes."""
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            QMessageBox.critical(self, "Error", f"Command failed with exit code: {exit_code}")
        self.run_button.setDisabled(False)

    def closeEvent(self, event):
        """Cleanup before exit."""
        self.process_manager.cleanup()
        # Cleanup analysis tabs
        for tab in self.tab_manager.analysis_tabs_map.values():
            if hasattr(tab, "shutdown"):
                tab.shutdown()
        super().closeEvent(event)
