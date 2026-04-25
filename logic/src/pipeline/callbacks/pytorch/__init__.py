"""
Lightning Callbacks for WSmart-Route.

PyTorch Lightning Callback implementations for training monitoring,
meta-learning updates, speed measurement, and model summaries.

Attributes:
    ModelSummaryCallback: Prints a rich table of model architecture details at training start.
    ReptileCallback: Outer-loop Reptile meta-learning update across tasks.
    SpeedMonitor: Logs forward/backward and data-loading times per step.
    TrainingDisplayCallback: Live terminal dashboard combining chart, metrics, and progress bars.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch import ModelSummaryCallback
    >>> cb = ModelSummaryCallback()
"""
