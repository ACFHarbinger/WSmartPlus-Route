// src/constants.ts
export const PROBLEM_TYPES = ["TSP", "VRP", "VRPP", "VRPTW", "OP", "PCTSP"];
export const DATA_DISTRIBUTIONS = {
  "Gamma 1": "gamma1",
  "Gamma 2": "gamma2",
  "Uniform": "uniform",
  "Lognormal": "lognormal",
  "All": "all",
};
export const DATA_DIST_PROBLEMS = ["VRP", "VRPP", "VRPTW"];
export const WASTE_TYPES = ["Plastic", "Paper", "Glass", "Organic", "Undifferentiated"];
export const COUNTY_AREAS = {
  "Rio Maior": "rio_maior",
  "Lisbon": "lisbon",
  "Porto": "porto",
  "riomaior": "rio_maior", // Alias from eval
};
export const DISTANCE_MATRIX_METHODS = {
  "Google Maps (GMaps)": "gmaps",
  "OSRM": "osrm",
  "Haversine": "haversine",
  "Local File": "local_file",
};
export const VERTEX_METHODS = {
  "Min-Max Normalization": "min_max",
  "Standardization": "standard",
  "None": "none",
};
export const EDGE_METHODS = {
  "K-Nearest Neighbors (KNN)": "knn",
  "Radius Graph": "radius",
  "None": "none",
};
export const DECODE_TYPES = ["greedy", "sample"];
export const SIMULATOR_TEST_POLICIES = {
  "Policy 1": "p1",
  "Policy 2": "p2",
  "Policy 3": "p3",
  "Policy 4": "p4",
  "Policy 5": "p5",
  "Policy 6": "p6",
  "Policy 7": "p7",
};
export const OPERATION_MAP = {
  "Add": "add",
  "Subtract": "sub",
  "Multiply": "mul",
  "Divide": "div",
  "Replace String": "replace_str",
};
export const FUNCTION_MAP = {
  "Mean": "mean",
  "Sum": "sum",
  "Median": "median",
  "Min": "min",
  "Max": "max",
};
export const BASELINES = ["critic", "exponential", "greedy", "rollout", "value"];
export const MODELS = {
  "Attention Model": "attn",
  "Pointer Network": "ptr",
};
export const ENCODERS = {
  "Graph Attention Encoder (GAT)": "gat",
  "Graph Convolutional Network (GCN)": "gcn",
  "Transformer": "transformer",
};
export const NORMALIZATION_METHODS = [
  "batch_norm",
  "instance_norm",
  "layer_norm",
  "none",
];
export const AGGREGATION_FUNCTIONS = {
  "Mean": "mean",
  "Sum": "sum",
  "Max": "max",
  "Min": "min",
  "Average": "mean", // Alias
};
export const ACTIVATION_FUNCTIONS = {
  "ReLU": "relu",
  "LeakyReLU": "leaky_relu",
  "GeLU": "gelu",
};
export const LR_SCHEDULERS = {
  "Step Learning Rate": "step",
  "Cosine Annealing": "cosine",
  "Lambda Learning Rate": "lambda",
};
export const OPTIMIZERS = {
  "Adam": "adam",
  "AdamW": "adamw",
  "Root Mean Square Propagation (RMSProp)": "rmsprop",
  "Stochastic Gradient Descent (SGD)": "sgd",
};
export const WB_MODES = ["online", "offline", "disabled"];
export const HOP_METHODS = {
  "Bayesian Optimization (BO)": "bo",
  "Distributed Evolutionary Algorithm (DEA)": "dea",
  "Hyperband Optimization (HBO)": "hbo",
  "Grid Search (GS)": "gs",
  "Differential Evolutionary Hyperband (DEHBO)": "dehbo",
  "Random Search (RS)": "rs",
};
export const HOP_METRICS = {
  "Validation Loss": "val_loss",
  "Reward": "reward",
  "Cost": "cost",
};
export const MRL_METHODS = {
  "Temporal Difference Learning (TDL)": "tdl",
  "Contextual Bandits (CB)": "cb",
  "Multi-Objective RL (MORL)": "morl",
  "Reward Weight Adjustment (RWA)": "rwa",
};
export const CB_EXPLORATION_METHODS = {
  "Upper Confidence Bound (UCB)": "ucb",
  "Epsilon Greedy": "epsilon_greedy",
  "Thompson Sampling": "thompson_sampling",
};
export const RWA_MODELS = {
  "Recurrent Neural Network (RNN)": "rnn",
  "Feed Forward Network (FFN)": "ffn",
  "Transformer": "transformer",
};
export const RWA_OPTIMIZERS = {
  "Root Mean Square Propagation (RMSProp)": "rmsprop",
  "Adam": "adam",
  "SGD": "sgd",
};
export const DECODE_STRATEGIES = {
  "Greedy": "greedy",
  "Sampling": "sample",
  "Beam Search": "bs",
};
export const SCRIPTS = {
  "setup_env": "Setup Environment",
  "gen_data": "Generate Data",
  "train": "Train Model",
  "hyperparam_optim": "Hyperparameter Optimization",
  "evaluation": "Model Evaluation",
  "test_sim": "Test Simulation",
  "slurm": "Slurm",
  "slim_slurm": "Slim Slurm",
};