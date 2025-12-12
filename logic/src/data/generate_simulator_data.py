import math
import numpy as np

from logic.src.pipeline.simulator.bins import Bins
from logic.src.utils.data_utils import load_focus_coords
from logic.src.utils.functions import get_path_until_string


def generate_wsr_data(problem_size, num_days, num_samples, area, waste_type, distribution, focus_graph, method):
    def _get_fill_gamma(num_samples, num_days, problem_size, gamma_option):
        def __set_distribution_param(size, param):
            param_len = len(param)
            if size == param_len:
                return param
            
            param = param * math.ceil(size / param_len)
            if size % param_len != 0:
                param = param[:param_len-size % param_len]
            return param

        if gamma_option == 0:
            alpha = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10]
            theta = [5, 2]
        elif gamma_option == 1:
            alpha = [2, 2, 2, 2, 2, 6, 6, 6, 6, 6]
            theta = [6, 4]
        elif gamma_option == 2:
            alpha = [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
            theta = [8, 6]
        else:
            assert gamma_option == 3
            alpha = [5, 2]
            theta = [10]

        k = __set_distribution_param(problem_size, alpha)
        th = __set_distribution_param(problem_size, theta)
        return np.random.gamma(k, th, size=(num_samples, num_days, problem_size))

    depot, loc, _, idx = load_focus_coords(problem_size, method, area, waste_type, focus_graph)
    assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]

    # Bin waste fill for all samples and days
    if 'gamma' in distribution:
        gamma_option = int(distribution[-1]) - 1
        waste = _get_fill_gamma(num_samples, num_days, problem_size, gamma_option)
    else:
        assert 'emp' in distribution
        fill_values = []
        data_dir = get_path_until_string(focus_graph, 'wsr_simulator')
        bins = Bins(problem_size, data_dir, sample_dist=distribution, area=area, indices=idx[0], grid=None)
        for _ in range(num_samples):
            fill_values.append(bins.stochasticFilling(n_samples=num_days, only_fill=True))
        waste = np.array(fill_values)
    return waste.tolist() if waste.shape[0] > 1 else waste[0].tolist()
