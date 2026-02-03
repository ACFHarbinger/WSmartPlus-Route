import pytest

from gui.src.tabs.evaluation.eval_decoding import EvalDecodingTab
from gui.src.tabs.evaluation.eval_problem import EvalProblemTab


class TestEvalTabInteractions:
    @pytest.fixture
    def problem_tab(self, qapp):
        tab = EvalProblemTab()
        yield tab
        tab.close()
        tab.deleteLater()
        qapp.processEvents()

    @pytest.fixture
    def decoding_tab(self, qapp):
        tab = EvalDecodingTab()
        yield tab
        tab.close()
        tab.deleteLater()
        qapp.processEvents()

    def test_problem_tab_toggle_focus(self, problem_tab):
        """Test the expand/collapse logic for Focus Graph section."""
        # Initial state: Hidden
        assert not problem_tab.is_focus_graph_visible
        assert problem_tab.focus_graph_container.isHidden()
        assert problem_tab.focus_graph_toggle_button.text() == "+"

        # Toggle ON
        problem_tab._toggle_focus_graph()
        assert problem_tab.is_focus_graph_visible
        assert not problem_tab.focus_graph_container.isHidden()
        assert problem_tab.focus_graph_toggle_button.text() == "-"

        # Toggle OFF
        problem_tab._toggle_focus_graph()
        assert not problem_tab.is_focus_graph_visible
        assert problem_tab.focus_graph_container.isHidden()

    def test_problem_tab_get_params(self, problem_tab):
        """Test parameter retrieval mechanics and mappings."""
        # Set values
        problem_tab.graph_size_input.setValue(100)

        # Area Mapping: "Rio Maior" -> "riomaior"
        # Since area_input is QLineEdit, user types arbitrary text?
        # Code: COUNTY_AREAS.get(self.area_input.text().strip(), "")
        # But wait, area_input is initialized with "riomaior".
        # If I look at the code: self.area_input = QLineEdit("riomaior")
        # And params uses COUNTY_AREAS.get(text, "")
        # But "riomaior" is NOT a key in COUNTY_AREAS (keys are Title Case).
        # Let's check constants/simulator.py: keys are "Rio Maior".
        # So default "riomaior" would fail lookup and return ""?
        # Let's verify this behavior.

        problem_tab.area_input.setText("Rio Maior")
        params = problem_tab.get_params()

        assert params["graph_size"] == 100
        assert params["area"] == "riomaior"

        # Distance Mapping
        problem_tab.distance_method_combo.setCurrentText("Google Maps (GMaps)")
        params = problem_tab.get_params()
        assert params["distance_method"] == "gmaps"

        # Test Focus Graph inclusion
        problem_tab.focus_graph_input.setText("/tmp/graph.pt")
        params = problem_tab.get_params()
        assert params["focus_graph"] == "/tmp/graph.pt"

    def test_decoding_tab_get_params(self, decoding_tab):
        """Test decoding parameters."""
        decoding_tab.decode_strategy_combo.setCurrentText("Sampling")
        decoding_tab.decode_type_combo.setCurrentText("Greedy")
        decoding_tab.width_input.setText("10 20")
        decoding_tab.softmax_temperature_input.setValue(1.5)

        params = decoding_tab.get_params()

        assert params["decode_strategy"] == "sampling"
        assert params["decode_type"] == "greedy"
        assert params["width"] == "10 20"
        assert params["softmax_temperature"] == 1.5
