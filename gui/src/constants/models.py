"""
Neural network model and encoding constants for the GUI.
"""

# Models
# TODO: Rename to STRATEGIES to match logic layer
DECODE_TYPES = ["greedy", "sampling"]

DECODE_STRATEGIES = {
    "Greedy": "greedy",
    "Sampling": "sampling",
    "Beam Search (BS)": "bs",
}

AGGREGATION_FUNCTIONS = {
    "Summation (sum)": "sum",
    "Average (avg)": "avg",
    "Maximum (max)": "max",
}

MODELS = {
    "Attention Model (AM)": "am",
    "Temporal Attention Model (TAM)": "tam",
    "Deep Decoder Attention Model (DDAM)": "ddam",
    "Pointer Network (PN)": "pn",
}

ENCODERS = {
    "Graph Attention (GAT)": "gat",
    "Graph Attention and Convolution (GAC)": "gac",
    "Transformer - Graph Convolutional Network (TGC)": "tgc",
    "Graph Convolutional Network (GCN)": "gcn",
    "Multi-Layer Perceptron (MLP)": "mlp",
}

NORMALIZATION_METHODS = ["instance", "layer", "batch", "group", "local_response"]

ACTIVATION_FUNCTIONS = {
    "Gaussian Error Linear Unit (GELU)": "gelu",
    "GELU-Tanh": "gelu_tanh",
    "Hyperbolic Tangent (Tanh)": "tanh",
    "Tanh Shrinkage (Tanhshrink)": "tanhshrink",
    "Mish": "mish",
    "Hard Shrinkage (Hardshrink)": "hardshrink",
    "Hard-Tanh": "hardtanh",
    "Hard-Swish": "hardswish",
    "Gated Linear Unit (GLU)": "glu",
    "Rectified Linear Unit (ReLU)": "relu",
    "Leaky-ReLU": "leakyrelu",
    "Sigmoid Linear Unit (SiLU)": "silu",
    "Exponential Linear Unit": "elu",
    "Scaled ELU (SELU)": "selu",
    "Continuously Differentiable Exponential Linear Unit (CELU)": "celu",
    "Parametric ReLU (PReLU)": "prelu",
    "Randomized Leaky ReLU (RReLU)": "rrelu",
    "Sigmoid": "sigmoid",
    "Log-Sigmoid": "logsigmoid",
    "Hard-Sigmoid": "hardsigmoid",
    "Threshold": "threshold",
    "Soft-Plus": "softplus",
    "Soft-Shrink": "softshrink",
    "Soft-Sign": "softsign",
}
