from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QGridLayout, QCheckBox,
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QComboBox, QSpinBox, QGroupBox, QScrollArea,
)
from ..app_definitions import TEST_MODULES


class TestSuiteTab(QWidget):
    """
    A QWidget representing the Test Suite configuration tab.
    """
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Use a QScrollArea for the main content to ensure responsiveness
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        # Call methods to create and store widget references
        self.create_test_selection_group()
        self.create_test_execution_group()
        self.create_information_commands_group()

        self.content_layout.addStretch(1) # Push content to the top
        scroll_area.setWidget(self.content_widget)
        self.layout.addWidget(scroll_area)

    def _add_grid_row(self, layout, row, label_text, widget):
        """Helper function to add a label and a widget to a QGridLayout."""
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        return row + 1

    def create_test_selection_group(self):
        """Creates the 'Test Selection' QGroupBox."""
        group_box = QGroupBox("🔍 Test Selection")
        grid = QGridLayout(group_box)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        row = 0

        # --module (QComboBox with multiselect simulation)
        self.module_combo = QComboBox() # Store reference
        self.module_combo.setEditable(False)
        self.module_combo.setToolTip("Specific test module(s) to run")
        self.module_combo.addItems(list(TEST_MODULES.keys()))
        self.module_combo.setCurrentIndex(-1) # No selection by default
        row = self._add_grid_row(grid, row, "Module(s):", self.module_combo)

        # --class (QLineEdit)
        self.class_input = QLineEdit() # Store reference
        self.class_input.setPlaceholderText("e.g., TestTrainCommand")
        self.class_input.setToolTip("Specific test class to run")
        row = self._add_grid_row(grid, row, "Test Class:", self.class_input)

        # --test (QLineEdit)
        self.method_input = QLineEdit() # Store reference
        self.method_input.setPlaceholderText("e.g., test_train_default_parameters")
        self.method_input.setToolTip("Specific test method to run")
        row = self._add_grid_row(grid, row, "Test Method:", self.method_input)

        # --keyword (QLineEdit)
        self.keyword_input = QLineEdit() # Store reference
        self.keyword_input.setToolTip("Run tests matching the given keyword expression")
        row = self._add_grid_row(grid, row, "Keyword:", self.keyword_input)

        # --markers (QLineEdit)
        self.markers_input = QLineEdit() # Store reference
        self.markers_input.setToolTip("Run tests matching the given marker expression")
        row = self._add_grid_row(grid, row, "Markers:", self.markers_input)

        self.content_layout.addWidget(group_box)

    def create_test_execution_group(self):
        """Creates the 'Test Execution Options' QGroupBox."""
        group_box = QGroupBox("⚙️ Test Execution Options")
        grid = QGridLayout(group_box)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        row = 0

        # Checkboxes for simple boolean flags (action='store_true')
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # -v, --verbose
        self.verbose_check = QCheckBox("Verbose Output") # Store reference
        self.verbose_check.setToolTip("Verbose output")
        checkbox_layout.addWidget(self.verbose_check)

        # --coverage
        self.coverage_check = QCheckBox("Coverage Report") # Store reference
        self.coverage_check.setToolTip("Run with coverage report")
        checkbox_layout.addWidget(self.coverage_check)

        # --ff, --failed-first
        self.ff_check = QCheckBox("Failed First") # Store reference
        self.ff_check.setToolTip("Run failed tests first")
        checkbox_layout.addWidget(self.ff_check)

        # -x, --exitfirst (action='store_const', const=1)
        self.exitfirst_check = QCheckBox("Exit on First Failure") # Store reference
        self.exitfirst_check.setToolTip("Exit on first failure (sets maxfail to 1)")
        checkbox_layout.addWidget(self.exitfirst_check)

        # -n, --parallel
        self.parallel_check = QCheckBox("Parallel") # Store reference
        self.parallel_check.setToolTip("Run tests in parallel (requires pytest-xdist)")
        checkbox_layout.addWidget(self.parallel_check)

        grid.addLayout(checkbox_layout, row, 0, 1, 2)
        row += 1

        # --maxfail (QSpinBox) - Note: If exitfirst is checked, maxfail is 1
        self.maxfail_spin = QSpinBox() # Store reference
        self.maxfail_spin.setRange(0, 1000)
        self.maxfail_spin.setValue(0)
        self.maxfail_spin.setToolTip("Exit after N failures (0 means no limit)")
        row = self._add_grid_row(grid, row, "Max Failures:", self.maxfail_spin)

        # --tb (QComboBox)
        self.tb_combo = QComboBox() # Store reference
        tb_choices = ['auto', 'long', 'short', 'line', 'native', 'no']
        self.tb_combo.addItems(tb_choices)
        self.tb_combo.setCurrentText('auto')
        self.tb_combo.setToolTip("Traceback print mode")
        row = self._add_grid_row(grid, row, "Traceback Mode:", self.tb_combo)

        # --capture (QComboBox)
        self.capture_combo = QComboBox() # Store reference
        capture_choices = ['auto', 'no', 'sys', 'fd']
        self.capture_combo.addItems(capture_choices)
        self.capture_combo.setCurrentText('auto')
        self.capture_combo.setToolTip("Capture mode for output")
        row = self._add_grid_row(grid, row, "Capture Mode:", self.capture_combo)

        self.content_layout.addWidget(group_box)

    def create_information_commands_group(self):
        """Creates the 'Information Commands' QGroupBox."""
        group_box = QGroupBox("Information and Configuration")
        grid = QGridLayout(group_box)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        row = 0

        # Checkboxes for simple boolean flags (action='store_true')
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # -l, --list
        self.list_check = QCheckBox("List Modules") # Store reference
        self.list_check.setToolTip("List all available test modules")
        checkbox_layout.addWidget(self.list_check)

        # --list-tests
        self.list_tests_check = QCheckBox("List Tests") # Store reference
        self.list_tests_check.setToolTip("List all tests in specified module(s) or all tests")
        checkbox_layout.addWidget(self.list_tests_check)

        grid.addLayout(checkbox_layout, row, 0, 1, 2)
        row += 1

        # --test-dir (QLineEdit)
        self.test_dir_input = QLineEdit('tests') # Store reference
        self.test_dir_input.setToolTip("Directory containing test files (default: tests)")
        row = self._add_grid_row(grid, row, "Test Directory:", self.test_dir_input)

        self.content_layout.addWidget(group_box)

    def get_params(self):
        """
        Extracts the current settings from the GUI widgets into a dictionary
        mimicking the output of argparse.
        """
        params = {}

        # Helper function to get text and replace empty strings with None
        def get_text_or_none(widget):
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                return text if text else None
            # For QComboBox used as single-select/placeholder
            elif isinstance(widget, QComboBox):
                text = widget.currentText().strip()
                return [text] if text else None
            return None
        
        # --- Test Selection ---
        # -m, --module (nargs='+') - We return a list of the selected module (or None)
        params['module'] = get_text_or_none(self.module_combo)
        
        # -c, --class
        params['test_class'] = get_text_or_none(self.class_input)
        
        # -t, --test
        params['test_method'] = get_text_or_none(self.method_input)
        
        # -k, --keyword
        params['keyword'] = get_text_or_none(self.keyword_input)
        
        # --markers
        params['markers'] = get_text_or_none(self.markers_input)

        # --- Test Execution Options ---
        # -v, --verbose (action='store_true')
        params['verbose'] = self.verbose_check.isChecked()

        # --coverage (action='store_true')
        params['coverage'] = self.coverage_check.isChecked()

        # --ff, --failed-first (action='store_true')
        params['failed_first'] = self.ff_check.isChecked()

        # -x, --exitfirst (action='store_const', const=1)
        # --maxfail (type=int)
        
        # If exitfirst is checked, it overrides maxfail to 1. Otherwise, use the spinbox value.
        if self.exitfirst_check.isChecked():
            params['maxfail'] = 1
        else:
            maxfail_value = self.maxfail_spin.value()
            # If the value is 0 (the default for no limit), argparse would typically get None 
            # if no value was passed, but here we explicitly store the int.
            params['maxfail'] = maxfail_value if maxfail_value > 0 else None 

        # --tb (choices)
        params['tb'] = self.tb_combo.currentText()

        # --capture (choices)
        params['capture'] = self.capture_combo.currentText()

        # -n, --parallel (action='store_true')
        params['parallel'] = self.parallel_check.isChecked()

        # --- Information Commands ---
        # -l, --list (action='store_true')
        params['list'] = self.list_check.isChecked()

        # --list-tests (action='store_true')
        params['list_tests'] = self.list_tests_check.isChecked()

        # --test-dir (str, default='tests')
        # This argument has a default, so we return the text even if empty, 
        # but since we initialized it to 'tests' it will rarely be empty.
        test_dir = self.test_dir_input.text().strip()
        params['test_dir'] = test_dir if test_dir else 'tests' # Re-apply default if user cleared it

        return params
