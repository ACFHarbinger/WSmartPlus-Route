from logic.src.utils.io import (
    find_single_input_values,
    process_dict_of_dicts,
    process_list_of_dicts,
)

class TestIOProcessing:
    """
    Tests for io/processing.py module (preserved from original test_utils.py).
    Duplicates logic in test_io.py but kept for backward compatibility request.
    """

    def test_process_dict_of_dicts_single_value(self):
        """Test processing a dict of dicts with single values."""
        data = {"policy1": {"km": 10.0, "waste": 50.0}}
        modified = process_dict_of_dicts(data, output_key="km", process_func=lambda x, y: x * 2, update_val=0)
        assert modified
        assert data["policy1"]["km"] == 20.0

    def test_process_dict_of_dicts_list_values(self):
        """Test processing a dict of dicts with list values."""
        data = {"policy1": {"km": [10.0, 20.0], "waste": 50.0}}
        modified = process_dict_of_dicts(data, output_key="km", process_func=lambda x, y: x + 5, update_val=0)
        assert modified
        assert data["policy1"]["km"] == [15.0, 25.0]

    def test_process_list_of_dicts(self):
        """Test processing a list of dicts."""
        data = [{"policy1": {"km": 10.0}}, {"policy2": {"km": 20.0}}]
        modified = process_list_of_dicts(data, output_key="km", process_func=lambda x, y: x / 2, update_val=0)
        assert modified
        assert data[0]["policy1"]["km"] == 5.0
        assert data[1]["policy2"]["km"] == 10.0

    def test_find_single_input_values(self):
        """Test finding single input values in nested dict."""
        data = {"policy1": {"day1": {"km": 100}}, "policy2": {"day1": {"km": 200}}}
        values = find_single_input_values(data, output_key="km")
        assert len(values) == 2
