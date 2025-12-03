from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QLineEdit, QSpinBox, QComboBox,
    QFormLayout, QVBoxLayout, QLabel, 
    QDoubleSpinBox,QScrollArea, QWidget,
    QHBoxLayout, QPushButton, QSizePolicy,
)
from src.gui.app_definitions import (
    VERTEX_METHODS, EDGE_METHODS,
    DISTANCE_MATRIX_METHODS, COUNTY_AREAS
)
from ...components import ClickableHeaderWidget


class EvalProblemTab(QWidget):
    """
    Tab for defining the problem instance characteristics and preprocessing methods.
    """
    def __init__(self):
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)
        
        # Ensure standard QFormLayout behavior (optional but good practice)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        form_layout.addRow(QLabel("<b>Problem Instance</b>"))

        # --graph_size
        self.graph_size_input = QSpinBox(minimum=1, maximum=500, value=50)
        form_layout.addRow("Graph Size:", self.graph_size_input)
        
        # --area
        self.area_input = QLineEdit('riomaior')
        form_layout.addRow("County Area:", self.area_input)
        
        # --waste_type
        self.waste_type_input = QLineEdit('plastic')
        form_layout.addRow("Waste Type:", self.waste_type_input)    

        # --------------------------------------------------------------------
        # --- Focus Graph (Custom Header) ---
        # --------------------------------------------------------------------
        # Create a container widget for the header using the custom clickable class
        self.focus_graph_header_widget = ClickableHeaderWidget(self._toggle_focus_graph)
        self.focus_graph_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        fg_header_layout = QHBoxLayout(self.focus_graph_header_widget)
        fg_header_layout.setContentsMargins(0, 0, 0, 0)
        fg_header_layout.setSpacing(5)

        # The main text (Standard QLabel)
        self.focus_graph_label = QLabel("<b>Focus Graph</b>")
        
        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.focus_graph_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.focus_graph_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # The clickable toggle button (only the +/- sign)
        self.focus_graph_toggle_button = QPushButton("+")
        self.focus_graph_toggle_button.setFlat(True)
        self.focus_graph_toggle_button.setFixedSize(QSize(20, 20))
        self.focus_graph_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.focus_graph_toggle_button.clicked.connect(self._toggle_focus_graph)
        
        # Add components to the header layout
        fg_header_layout.addWidget(self.focus_graph_label)
        fg_header_layout.addStretch()
        fg_header_layout.addWidget(self.focus_graph_toggle_button)
        
        # Add the header widget to the main layout, which should put it in the Field column
        form_layout.addRow(self.focus_graph_header_widget)

        # Create a container for the collapsible content
        self.focus_graph_container = QWidget()
        focus_graph_layout = QFormLayout(self.focus_graph_container)
        focus_graph_layout.setContentsMargins(0, 5, 0, 0) 

        # Add widgets to the container's layout
        # --focus_graph
        self.focus_graph_input = QLineEdit()
        self.focus_graph_input.setPlaceholderText("Path to focus graph file")
        focus_graph_layout.addRow("Focus Graph Path:", self.focus_graph_input)

        # --focus_size
        self.focus_size_input = QSpinBox(minimum=0, maximum=1000, value=0)
        focus_graph_layout.addRow("Number of Focus Graphs:", self.focus_size_input)
        
        # The collapsible content container is added here
        form_layout.addWidget(self.focus_graph_container)

        # Initialize state: hidden (must match the default style above)
        self.is_focus_graph_visible = False
        self.focus_graph_container.hide()

        # --------------------------------------------------------------------
        form_layout.addRow(QLabel("<b>Preprocessing Methods</b>"))

        # --distance_method
        self.distance_method_combo = QComboBox()
        self.distance_method_combo.addItems(DISTANCE_MATRIX_METHODS.keys())
        self.distance_method_combo.setCurrentText('Google Maps (GMaps)')
        form_layout.addRow("Distance Method:", self.distance_method_combo)

        # --vertex_method
        self.vertex_method_combo = QComboBox()
        self.vertex_method_combo.addItems(VERTEX_METHODS.keys())
        self.vertex_method_combo.setCurrentText('Min-Max Normalization')
        form_layout.addRow("Vertex Method:", self.vertex_method_combo)
        
        # --edge_threshold (Used QDoubleSpinBox as per previous constraint)
        self.edge_threshold_input = QDoubleSpinBox()
        self.edge_threshold_input.setRange(0.0, 1.0)
        self.edge_threshold_input.setSingleStep(0.01)
        self.edge_threshold_input.setDecimals(2)
        self.edge_threshold_input.setValue(1.0)
        form_layout.addRow("Edge Threshold (0.0 to 1.0):", self.edge_threshold_input)

        # --edge_method
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems(EDGE_METHODS.keys())
        self.edge_method_combo.setCurrentIndex(0)
        form_layout.addRow("Edge Method:", self.edge_method_combo)

        scroll_area.setWidget(content)
        QVBoxLayout(self).addWidget(scroll_area)

    def _toggle_focus_graph(self):
        """
        Toggles the visibility of the Focus Graph input fields and updates the +/- sign 
        and the header border styling.
        """
        if self.is_focus_graph_visible:
            self.focus_graph_container.hide()
            self.focus_graph_toggle_button.setText("+")
            
            # Apply dark grey border to the QLabel when collapsed
            self.focus_graph_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.focus_graph_container.show()
            self.focus_graph_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.focus_graph_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_focus_graph_visible = not self.is_focus_graph_visible

    def get_params(self):
        params = {
            "graph_size": self.graph_size_input.value(),
            "area": COUNTY_AREAS.get(self.area_input.text().strip(), ""),
            "waste_type": self.waste_type_input.text().strip().lower(),
            "focus_graph": self.focus_graph_input.text().strip() or None,
            "focus_size": self.focus_size_input.value(),
            
            # Numeric/Float values
            "edge_threshold": self.edge_threshold_input.value(),
            
            # ComboBox values
            "distance_method": DISTANCE_MATRIX_METHODS.get(self.distance_method_combo.currentText(), ""),
            "vertex_method": VERTEX_METHODS.get(self.vertex_method_combo.currentText(), ""),
            "edge_method": EDGE_METHODS.get(self.edge_method_combo.currentText(), "") or None,
        }
        return {k: v for k, v in params.items() if v is not None}
