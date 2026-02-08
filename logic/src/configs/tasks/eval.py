"""
Eval Config module.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..envs.graph import GraphConfig
from ..envs.objective import ObjectiveConfig
from ..models.decoding import DecodingConfig
from ..policies.neural import NeuralConfig


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        datasets: Filename of the dataset(s) to evaluate.
        overwrite: Set true to overwrite.
        output_filename: Name of the results file to write.
        val_size: Number of instances used for reporting validation performance.
        offset: Offset where to start in dataset.
        eval_batch_size: Batch size to use during (baseline) evaluation.
        decoding: Decoding configuration.
        model: Model configuration.
        seed: Random seed to use.
        data_distribution: Data distribution of the dataset.
        no_cuda: Disable CUDA.
        no_progress_bar: Disable progress bar.
        compress_mask: Compress mask into long.
        max_calc_batch_size: Size for subbatches.
        results_dir: Name of evaluation results directory.
        multiprocessing: Use multiprocessing to parallelize over multiple GPUs.
        graph: Graph/instance configuration.
        reward: Objective/reward configuration.
        problem: Problem to evaluate.
    """

    datasets: Optional[List[str]] = None
    overwrite: bool = False
    output_filename: Optional[str] = None
    val_size: int = 12_800
    offset: int = 0
    eval_batch_size: int = 256
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    policy: NeuralConfig = field(default_factory=NeuralConfig)
    seed: int = 42
    data_distribution: Optional[str] = None
    no_cuda: bool = False
    no_progress_bar: bool = False
    compress_mask: bool = False
    max_calc_batch_size: int = 12_800
    results_dir: str = "results"
    multiprocessing: bool = False
    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    problem: str = "cwcvrp"
