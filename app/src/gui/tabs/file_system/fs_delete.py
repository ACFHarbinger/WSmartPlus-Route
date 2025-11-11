from PySide6.QtWidgets import (
    QCheckBox, QLabel, QScrollArea,
    QWidget, QLineEdit, QFormLayout, 
    QVBoxLayout, QGroupBox, QPushButton
)
from ...styles import START_RED_STYLE


class FileSystemDeleteTab(QWidget):
    """
    GUI tab for Delete settings, focused on deleting file system entries.
    The contents are now wrapped in a QScrollArea to make them scrollable.
    """
    def __init__(self):
        super().__init__()
        # 1. Create a container widget for all content
        scrollable_content = QWidget()
        # 2. Create the layout that will hold all the settings and be scrollable
        content_layout = QVBoxLayout(scrollable_content)
        
        # Add the title to the scrollable content layout
        content_layout.addWidget(QLabel("<h2>Delete File System Entries</h2>"))

        # --- Directory Path Group ---
        path_group = QGroupBox()
        path_layout = QFormLayout(path_group)
        
        self.log_dir_input = QLineEdit('logs')
        path_layout.addRow("Train Log Directory:", self.log_dir_input)
        
        self.output_dir_input = QLineEdit('model_weights')
        path_layout.addRow("Output Models Directory:", self.output_dir_input)
        
        self.data_dir_input = QLineEdit('datasets')
        path_layout.addRow("Datasets Directory:", self.data_dir_input)
        
        self.eval_dir_input = QLineEdit('results')
        path_layout.addRow("Evaluation Results Directory:", self.eval_dir_input)

        self.test_dir_input = QLineEdit('output')
        path_layout.addRow("WSR Test Output Directory:", self.test_dir_input)

        self.test_checkpoint_dir_input = QLineEdit('temp')
        path_layout.addRow("WSR Checkpoint Directory:", self.test_checkpoint_dir_input)
        
        content_layout.addWidget(path_group)
        
        # --- Deletion Flags Group ---
        flag_group = QGroupBox()
        flag_layout = QFormLayout(flag_group)
        flag_layout.addRow(QLabel('<span style="font-weight: 600;">Boolean Flags</span>'))
        
        # Helper dictionary for flags that use action="store_false"
        self.flags_store_false = {}
        
        # --wandb (action="store_false" -> default True)
        self.wandb_check = QCheckBox("Keep Weights and Biases Logs (Uncheck to Delete)")
        self.wandb_check.setChecked(True)
        self.flags_store_false['wandb'] = self.wandb_check
        flag_layout.addRow("WandB Logs:", self.wandb_check)

        # --log (action="store_false" -> default True)
        self.log_check = QCheckBox("Keep Train Logs (Uncheck to Delete)")
        self.log_check.setChecked(True)
        self.flags_store_false['log'] = self.log_check
        flag_layout.addRow("Train Logs:", self.log_check)
        
        # --output (action="store_false" -> default True)
        self.output_check = QCheckBox("Keep Model Weights (Uncheck to Delete)")
        self.output_check.setChecked(True)
        self.flags_store_false['output'] = self.output_check
        flag_layout.addRow("Model Weights:", self.output_check)
        
        flag_layout.addRow(QLabel("<hr>"))
        
        # Helper dictionary for flags that use action="store_true"
        self.flags_store_true = {}

        # --data (action="store_true" -> default False)
        self.data_check = QCheckBox("Delete Generated Datasets")
        self.flags_store_true['data'] = self.data_check
        flag_layout.addRow("Datasets:", self.data_check)
        
        # --eval (action="store_true" -> default False)
        self.eval_check = QCheckBox("Delete Evaluation Results")
        self.flags_store_true['eval'] = self.eval_check
        flag_layout.addRow("Evaluation Results:", self.eval_check)

        # --test_sim (action="store_true" -> default False)
        self.test_sim_check = QCheckBox("Delete WSR Simulator Output")
        self.flags_store_true['test_sim'] = self.test_sim_check
        flag_layout.addRow("Simulator Output:", self.test_sim_check)
        
        # --test_sim_checkpoint (action="store_true" -> default False)
        self.test_sim_checkpoint_check = QCheckBox("Delete Simulator Checkpoints")
        self.flags_store_true['test_sim_checkpoint'] = self.test_sim_checkpoint_check
        flag_layout.addRow("Simulator Checkpoints:", self.test_sim_checkpoint_check)
        
        # --cache (action="store_true" -> default False)
        self.cache_check = QCheckBox("Delete Cache Directories")
        self.flags_store_true['cache'] = self.cache_check
        flag_layout.addRow("Cache Directories:", self.cache_check)

        # --- Separator and Preview Checkbox ---
        flag_layout.addRow(QLabel("<hr>"))

        # --delete_preview (action='store_true' -> default False)
        self.preview_check = QPushButton("Preview Delete")
        self.preview_check.setCheckable(True)
        self.preview_check.setChecked(False)
        self.preview_check.setStyleSheet(START_RED_STYLE)
        flag_layout.addRow("Verify changes before deleting:", self.preview_check)
        
        content_layout.addWidget(flag_group)
        content_layout.addStretch() # Ensure content is pushed to the top

        # 3. Create the QScrollArea and set the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) # Important: allows the content to take up available space
        scroll_area.setWidget(scrollable_content)

        # 4. Set the final layout of the tab (self) to include only the scroll area
        tab_layout = QVBoxLayout(self)
        tab_layout.addWidget(scroll_area)


    def get_params(self):
        """Extracts settings into a dictionary mimicking argparse output."""
        params = {
            # Directory paths
            "log_dir": self.log_dir_input.text().strip(),
            "output_dir": self.output_dir_input.text().strip(),
            "data_dir": self.data_dir_input.text().strip(),
            "eval_dir": self.eval_dir_input.text().strip(),
            "test_dir": self.test_dir_input.text().strip(),
            "test_checkpoint_dir": self.test_checkpoint_dir_input.text().strip(),
        }
        
        # Handle action="store_false" flags (default=True). 
        # Argparse includes the flag if the value should be False.
        # Here: If the box is UNCHECKED (value=False), the deletion action is requested.
        for name, checkbox in self.flags_store_false.items():
            if not checkbox.isChecked():
                params[name] = False
        
        # Handle action="store_true" flags (default=False).
        # Argparse includes the flag if the value should be True.
        # Here: If the box is CHECKED (value=True), the deletion action is requested.
        for name, checkbox in self.flags_store_true.items():
            if checkbox.isChecked():
                params[name] = True
        
        # Handle --delete_preview (action='store_false' -> default True)
        # If button is UNCHECKED, the user wants to PERFORM the delete (delete_preview=False).
        if not self.preview_check.isChecked():
            params['delete_preview'] = False
                
        return params
