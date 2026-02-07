
import pytest
from logic.src.utils.data.data_utils import load_area_and_waste_type_params

class TestDataUtils:
    def test_load_area_and_waste_type_params_known(self):
        # Rio Maior, Paper
        # vehicle_capacity = 4000L. bin_vol=2.5. density=21.
        # capacity % = (4000 / (2.5 * 21)) * 100 = 7619.04...
        cap, rev, dens, exp, vol = load_area_and_waste_type_params("Rio Maior", "paper")

        assert dens == 21.0
        assert vol == 2.5
        assert exp == 1.0
        assert cap > 0

    def test_load_area_and_waste_type_params_error(self):
        with pytest.raises(AssertionError):
            load_area_and_waste_type_params("Rio Maior", "uranium")
