"""
Sim Config module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ..envs.graph import GraphConfig


@dataclass
class SimConfig:
    """Simulation configuration.

    Attributes:
        policies: Name of the policy(ies) to test on the WSR simulator.
        data_distribution: Distribution to generate the bins daily waste fill.
        problem: The problem the model was trained to solve.
        days: Number of days to run the simulation for.
        seed: Random seed.
        output_dir: Name of WSR simulator test output directory.
        checkpoint_dir: Name of WSR simulator test runs checkpoint directory.
        checkpoint_days: Number of days interval to save simulation checkpoints.
        log_level: Logging level for the system logger.
        log_file: Path to the system log file.
        n_samples: Number of simulation samplings for each policy.
        resume: Resume testing.
        n_vehicles: Number of vehicles.
        noise_mean: Mean of Gaussian noise to inject into observed bin levels.
        noise_variance: Variance of Gaussian noise to inject into observed bin levels.
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
        waste_filepath: Path to the file to read the waste fill for each day from.
        graph: Graph/instance configuration.
        reward: Objective/reward configuration.
        data_dir: Directory containing the simulation data.
    """

    policies: Optional[List[str]] = None
    data_distribution: str = "gamma1"
    problem: str = "vrpp"
    days: int = 31
    seed: int = 42
    output_dir: str = "output"
    checkpoint_dir: str = "temp"
    checkpoint_days: int = 0
    log_level: str = "INFO"
    log_file: str = f"logs/simulations/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    n_samples: int = 1
    resume: bool = False

    cpu_cores: int = 0
    n_vehicles: int = 1
    waste_filepath: Optional[str] = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    noise_mean: float = 0.0
    noise_variance: float = 0.0
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
    data_dir: Optional[str] = None
