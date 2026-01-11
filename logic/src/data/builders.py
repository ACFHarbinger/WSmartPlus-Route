import numpy as np

from logic.src.utils.definitions import MAX_WASTE
from logic.src.utils.functions import get_path_until_string
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.processor import process_coordinates


class VRPInstanceBuilder:
    """
    Builder pattern for creating Vehicle Routing Problem (VRP) instances.

    This class provides a fluent interface to configure and generate VRP datasets
    with various parameters such as problem size, distribution, area, and waste type.
    """
    def __init__(self):
        self._dataset_size = 10
        self._problem_size = 20
        self._waste_type = None
        self._distribution = "gamma1"
        self._area = "Rio Maior"
        self._focus_graph = None
        self._focus_size = 0
        self._method = None
        self._num_days = 1
        self._problem_name = None
        self._noise_mean = 0.0
        self._noise_variance = 0.0

    def set_dataset_size(self, size: int):
        """Sets the number of instances to generate."""
        self._dataset_size = size
        return self

    def set_problem_size(self, size: int):
        """Sets the number of nodes (graph size) for the problem."""
        self._problem_size = size
        return self

    def set_waste_type(self, waste_type: str):
        """Sets the type of waste (e.g., 'plastic', 'paper')."""
        self._waste_type = waste_type
        return self

    def set_distribution(self, distribution: str):
        """Sets the data distribution for generating waste levels."""
        self._distribution = distribution
        return self

    def set_area(self, area: str):
        """Sets the geographical area for the problem instance."""
        self._area = area
        return self

    def set_focus_graph(self, focus_graph: str = None, focus_size: int = 0):
        """Sets parameters for focusing on a specific subgraph."""
        self._focus_graph = focus_graph
        self._focus_size = focus_size
        return self

    def set_method(self, method: str):
        """Sets the method used for vertex generation/selection."""
        self._method = method
        return self

    def set_num_days(self, num_days: int):
        """Sets the number of simulation days."""
        self._num_days = num_days
        return self
    
    def set_problem_name(self, problem_name: str):
        """Sets the name of the problem (e.g., 'vrpp', 'wcvrp')."""
        self._problem_name = problem_name
        return self

    def set_noise(self, mean: float, variance: float):
        """Sets the mean and variance for noise injection."""
        self._noise_mean = mean
        self._noise_variance = variance
        return self

    def build(self):
        """
        Generates the dataset based on configured parameters.

        Returns:
            list: A list of problem instances, where each instance is a tuple containing:
                  - depot (list): Coordinates of the depot.
                  - loc (list): Coordinates of customer nodes.
                  - waste (list): Waste levels (or demand) for nodes.
                  - max_waste (float): Maximum capacity or waste limit.
        """
        if self._focus_graph is not None:
            assert self._focus_size > 0, "Focus size must be positive when using focus graph"
            
            # Load focus coordinates
            depot, loc, mm_arr, idx = load_focus_coords(
                self._problem_size, self._method, self._area, 
                self._waste_type, self._focus_graph, self._focus_size
            )
            
            # Generate remaining random coordinates
            remaining_coords_size = self._dataset_size - self._focus_size
            if remaining_coords_size > 0:
                random_coords = np.random.uniform(
                    mm_arr[0], mm_arr[1], 
                    size=(remaining_coords_size, self._problem_size + 1, mm_arr.shape[-1])
                )
                depots, locs = process_coordinates(random_coords, self._method, col_names=None)
                depot = np.concatenate((depot, depots))
                loc = np.concatenate((loc, locs))
            
            assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]
            
            # Set up bins if using empirical distribution
            bins = None
            if self._distribution == 'emp':
                data_dir = get_path_until_string(self._focus_graph, 'wsr_simulator')
                bins = Bins(
                    self._problem_size, data_dir, sample_dist=self._distribution, 
                    area=self._area, indices=idx[0], grid=None, waste_type=self._waste_type
                )
        else:
            bins = None
            coord_size = 2 if self._method != 'triple' else 3
            depot = np.random.uniform(size=(self._dataset_size, coord_size))
            loc = np.random.uniform(size=(self._dataset_size, self._problem_size, coord_size))

        # Generate waste/fill values over days
        fill_values = []
        # For waste generation, we pass the tuple of (depot, loc)
        coords = (depot, loc)
        
        for _ in range(self._num_days):
            waste = generate_waste_prize(
                self._problem_size, self._distribution, coords, self._dataset_size, bins
            )
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)
        
        # Transpose to (dataset_size, num_days, problem_size)
        fill_values = np.transpose(np.array(fill_values), (1, 0, 2))
        
        # Construct the output list
        if self._problem_name == 'swcvrp':
            # SWCVRP Case: Generate Noisy Waste
            real_waste_list = fill_values.tolist()
            
            # Generate Noise
            noise = np.random.normal(self._noise_mean, np.sqrt(self._noise_variance), fill_values.shape)
            noisy_fill_values = np.clip(fill_values + noise, 0, MAX_WASTE)
            noisy_waste_list = noisy_fill_values.tolist()
            
            return list(zip(
                depot.tolist(),
                loc.tolist(),
                real_waste_list,
                noisy_waste_list,
                np.full(self._dataset_size, MAX_WASTE).tolist()
            ))
        else:
            # Standard WCVRP/VRPP Case
            waste_list = fill_values.tolist()
            
            return list(zip(
                depot.tolist(),
                loc.tolist(),
                waste_list,
                np.full(self._dataset_size, MAX_WASTE).tolist()
            ))
