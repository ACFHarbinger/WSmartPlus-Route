from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SimConfig:
    """Simulation configuration.

    Attributes:
        policies: Name of the policy(ies) to test on the WSR simulator.
        gate_prob_threshold: Probability threshold for gating decisions.
        mask_prob_threshold: Probability threshold for mask decisions.
        data_distribution: Distribution to generate the bins daily waste fill.
        problem: The problem the model was trained to solve.
        size: The size of the problem graph.
        days: Number of days to run the simulation for.
        seed: Random seed.
        output_dir: Name of WSR simulator test output directory.
        checkpoint_dir: Name of WSR simulator test runs checkpoint directory.
        checkpoint_days: Number of days interval to save simulation checkpoints.
        log_level: Logging level for the system logger.
        log_file: Path to the system log file.
        cpd: Save checkpoint every n days.
        n_samples: Number of simulation samplings for each policy.
        resume: Resume testing.
        n_vehicles: Number of vehicles.
        area: County area of the bins locations.
        waste_type: Type of waste bins selected for the optimization problem.
        bin_idx_file: File with the indices of the bins to use in the simulation.
        decode_type: Decode type, greedy or sampling.
        temperature: Softmax temperature.
        edge_threshold: How many of all possible edges to consider.
        edge_method: Method for getting edges.
        vertex_method: Method to transform vertex coordinates.
        distance_method: Method to compute distance matrix.
        dm_filepath: Path to the file to read/write the distance matrix from/to.
        waste_filepath: Path to the file to read the waste fill for each day from.
        noise_mean: Mean of Gaussian noise to inject into observed bin levels.
        noise_variance: Variance of Gaussian noise to inject into observed bin levels.
        run_tsp: Activate fast_tsp for all policies.
        two_opt_max_iter: Maximum number of 2-opt iterations.
        cache_regular: Deactivate caching for policy regular (Default True).
        no_cuda: Disable CUDA.
        no_progress_bar: Disable progress bar.
        server_run: Simulation will be executed in a remote server.
        env_file: Name of the file that contains the environment variables.
        gplic_file: Name of the file that contains the license to use for Gurobi.
        hexlic_file: Name of the file that contains the license to use for Gurobi.
        symkey_name: Name of the cryptographic key used to access the API keys.
        gapik_file: Name of the file that contains the key to use for the Google API.
        real_time_log: Activate real time results window.
        stats_filepath: Path to the file to read the statistics from.
        model_path: Path to the directory where the model(s) is/are stored.
        config_path: Path to the YAML/XML configuration file(s).
        w_length: Weight for length in cost function.
        w_waste: Weight for waste in cost function.
        w_overflows: Weight for overflows in cost function.
        data_dir: Directory containing the simulation data.
    """

    policies: Optional[List[str]] = None
    gate_prob_threshold: float = 0.5
    mask_prob_threshold: float = 0.5
    data_distribution: str = "gamma1"
    problem: str = "vrpp"
    size: int = 50
    days: int = 31
    seed: int = 42
    output_dir: str = "output"
    checkpoint_dir: str = "temp"
    checkpoint_days: int = 0
    log_level: str = "INFO"
    log_file: str = "logs/simulation.log"
    cpd: int = 5
    n_samples: int = 1
    resume: bool = False

    cpu_cores: int = 0
    n_vehicles: int = 1
    area: str = "riomaior"
    waste_type: str = "plastic"
    bin_idx_file: Optional[str] = None
    decode_type: str = "greedy"
    temperature: float = 1.0
    edge_threshold: str = "0"
    edge_method: Optional[str] = None
    vertex_method: str = "mmn"
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    waste_filepath: Optional[str] = None
    noise_mean: float = 0.0
    noise_variance: float = 0.0
    run_tsp: bool = False
    two_opt_max_iter: int = 0
    cache_regular: bool = True
    no_cuda: bool = False
    no_progress_bar: bool = False
    server_run: bool = False
    env_file: str = "vars.env"
    gplic_file: Optional[str] = None
    hexlic_file: Optional[str] = None
    symkey_name: Optional[str] = None
    gapik_file: Optional[str] = None
    real_time_log: bool = False
    stats_filepath: Optional[str] = None
    model_path: Optional[Dict[str, str]] = None
    config_path: Optional[Dict[str, str]] = None
    w_length: float = 1.0
    w_waste: float = 1.0
    w_overflows: float = 1.0
    data_dir: Optional[str] = None
