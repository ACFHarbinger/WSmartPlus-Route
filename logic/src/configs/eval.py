"""
Eval Config module.
"""

from dataclasses import dataclass
from typing import List, Optional


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
        decode_type: Decode type, greedy or sampling.
        width: Sizes of beam to use for beam search (or number of samples for sampling).
        decode_strategy: Beam search (bs), Sampling (sample) or Greedy (greedy).
        softmax_temperature: Softmax temperature (sampling or bs).
        model: Model filename.
        seed: Random seed to use.
        data_distribution: Data distribution of the dataset.
        no_cuda: Disable CUDA.
        no_progress_bar: Disable progress bar.
        compress_mask: Compress mask into long.
        max_calc_batch_size: Size for subbatches.
        results_dir: Name of evaluation results directory.
        multiprocessing: Use multiprocessing to parallelize over multiple GPUs.
        num_loc: The number of customer locations (excluding depot).
        area: County area of the bins locations.
        waste_type: Type of waste bins selected for the optimization problem.
        focus_graph: Path to the file with the coordinates of the graph to focus on.
        focus_size: Number of focus graphs to include in the training data.
        edge_threshold: How many of all possible edges to consider.
        edge_method: Method for getting edges ('dist'|'knn').
        distance_method: Method to compute distance matrix.
        dm_filepath: Path to the distance matrix file.
        vertex_method: Method to transform vertex coordinates.
        w_length: Weight for length in cost function.
        w_waste: Weight for waste in cost function.
        w_overflows: Weight for overflows in cost function.
        problem: Problem to evaluate.
        encoder: Encoder to use.
        load_path: Path to load model parameters and optimizer state from.
    """

    datasets: Optional[List[str]] = None
    overwrite: bool = False
    output_filename: Optional[str] = None
    val_size: int = 12_800
    offset: int = 0
    eval_batch_size: int = 256
    decode_type: str = "greedy"
    width: Optional[List[int]] = None
    decode_strategy: Optional[str] = None
    softmax_temperature: float = 1.0
    model: Optional[str] = None
    seed: int = 42
    data_distribution: Optional[str] = None
    no_cuda: bool = False
    no_progress_bar: bool = False
    compress_mask: bool = False
    max_calc_batch_size: int = 12_800
    results_dir: str = "results"
    multiprocessing: bool = False
    num_loc: int = 50
    area: str = "riomaior"
    waste_type: str = "plastic"
    focus_graph: Optional[str] = None
    focus_size: int = 0
    edge_threshold: str = "0"
    edge_method: Optional[str] = None
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    vertex_method: str = "mmn"
    w_length: float = 1.0
    w_waste: float = 1.0
    w_overflows: float = 1.0
    problem: str = "cwcvrp"
    encoder: str = "gat"
    load_path: Optional[str] = None
