# Training
OPTIMIZERS = {
    "Adaptive Moment Estimation (Adam)": "adam",
    "Adam with Infinity Norm (Adamax)": "adamax",
    "Adam with Weight Decay (AdamW)": "adamw",
    "Rectified Adam (RAdam)": "radam",
    "Nesterov-accelerated Adam (NAdam)": "nadam",
    "Sparse-Adam (SAdam)": "sadam",
    "Adaptive Delta Method (AdaDelta)": "adadelta",
    "Adaptive Gradient Algorithm (AdaGrad)": "adagrad",
    "Root Mean Square Propagation (RMSProp)": "rmsprop",
    "Resilient Backpropagation (RProp)": "rprop",
    "Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS)": "lbfgs",
    "Averaged SGD (ASGD)": "asgd",
    "Stochastic Gradient Descent (SGD)": "sgd",
}

LR_SCHEDULERS = {
    "Lambda LR": "lambda",
    "Exponential LR": "exp",
    "Multiplicative LR": "mult",
    "Step LR": "step",
    "Constant LR": "const",
    "Polynomial LR": "poly",
    "Multi-Step LR": "multistep",
    "Cosine Annealing LR": "cosan",
    "Linear LR": "linear",
    "Cosine Annealing LR with Warm Restarts": "cosanwr",
    "Reduce LR on Plateau": "plateau",
}

BASELINES = ["Rollout", "Critic", "Exponential"]
