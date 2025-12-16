# Problems and Data
PROBLEM_TYPES = ["TSP", "VRP", "PCTSP", "VRPP", "WCRP", "OP", "PDP"]

DATA_DIST_PROBLEMS = ['VRPP', 'OP', 'WCRP', 'WCVRP']

DATA_DISTRIBUTIONS = {
    "Empty": "empty", 
    "Gamma 1": "gamma1", 
    "Gamma 2": "gamma2", 
    "Gamma 3": "gamma3", 
    "Empirical": "emp", 
    "Constant": "const", 
    "Uniform": "unif", 
    "Distance": "dist"
}

VERTEX_METHODS = {
    "Min-Max Normalization (MMN)": "mmn",
    "3D Cartesian Coordinates (C3D)": "c3d",
    "4D Spherical Coordinates (S4D)": "s4d",
    "Mean Normalization (MUN)": "mun",
    "Standardization using Mean and Standard Deviation (SMSD)": "smsd",
    "Equidistant Cylindrical Projection (ECP)": "ecp",
    "Universal Transverse Mercator Projection (UTMP)": "utmp",
    "World Mercator Projection (WMP)": "wmp",
    "Haversine Distance Projection (HDP)": "hdp",
}

EDGE_METHODS = {
    '': None,
    'Node Distance': 'dist',
    'K-Nearest Neighbors': 'knn',
}

# Simulator Data
COUNTY_AREAS = {
    "Rio Maior": "riomaior",
    "Figueira da Foz": "figdafoz",
    "Mixtura de Freguesias": "mixrmbac"    
}

WASTE_TYPES = ['', 'Paper', 'Plastic', 'Glass']

# Test Simulator
SIMULATOR_TEST_POLICIES = {
    "Attention Model": "am", 
    "Attention Model with Graph Connections": "amgc", 
    "TransformerGCN Model": "transgcn",
    "Deep Decoder Attention Model": "ddam", 
    "Gurobi VRPP Solver": "gurobi_vrpp", 
    "Hexaly VRPP Solver": "hexaly_vrpp", 
    "Look-Ahead Policy": "policy_look_ahead", 
    "Look-Ahead VRPP Policy": "policy_look_ahead_vrpp", 
    "Look-Ahead SANS Policy": "policy_look_ahead_sans",
    "Last Minute and Path Policy": "policy_last_minute_and_path", 
    "Last Minute Policy": "policy_last_minute", 
    "Regular Policy": "policy_regular"
}

DISTANCE_MATRIX_METHODS = {
    'Google Maps (GMaps)': 'gmaps',
    'Open Street Maps (OSM)': 'osm',
    'Geo-Pandas Distance (GPD)': 'gpd',
    'Geodesic Distance (GdsC)': 'gdsc',
    'Haversine Distance (HsD)': 'hsd',
    'Original Distance (OgD)': 'ogd'
}

# Models
DECODE_TYPES = ['greedy', 'sampling']

DECODE_STRATEGIES = {
    'Greedy': 'greedy', 
    'Sampling': 'sampling',
    'Beam Search (BS)': 'bs'
}

AGGREGATION_FUNCTIONS = {
    'Summation (sum)': 'sum', 
    'Average (avg)': 'avg', 
    'Maximum (max)': 'max'
}

MODELS = {
    'Attention Model (AM)': 'am',
    'Temporal Attention Model (TAM)': 'tam',
    'Deep Decoder Attention Model (DDAM)': 'ddam',
    'Pointer Network (PN)': 'pn'
}

ENCODERS = {
    'Graph Attention (GAT)': 'gat',
    'Graph Attention and Convolution (GAC)': 'gac',
    'Transformer - Graph Convolutional Network (TGC)': 'tgc',
    'Graph Convolutional Network (GCN)': 'gcn',
    'Multi-Layer Perceptron (MLP)': 'mlp'
}

NORMALIZATION_METHODS = ['instance', 'layer', 'batch', 'group', 'local_response']

ACTIVATION_FUNCTIONS = {
    'Gaussian Error Linear Unit (GELU)': 'gelu',
    'GELU-Tanh': 'gelu_tanh',
    'Hyperbolic Tangent (Tanh)': 'tanh',
    'Tanh Shrinkage (Tanhshrink)': 'tanhshrink',
    'Mish': 'mish',
    'Hard Shrinkage (Hardshrink)': 'hardshrink',
    'Hard-Tanh': 'hardtanh',
    'Hard-Swish': 'hardswish',
    'Gated Linear Unit (GLU)': 'glu',
    'Rectified Linear Unit (ReLU)': 'relu',
    'Leaky-ReLU': 'leakyrelu',
    'Sigmoid Linear Unit (SiLU)': 'silu',
    'Exponential Linear Unit': 'elu',
    'Scaled ELU (SELU)': 'selu',
    'Continuously Differentiable Exponential Linear Unit (CELU)': 'celu',
    'Parametric ReLU (PReLU)': 'prelu',
    'Randomized Leaky ReLU (RReLU)': 'rrelu',
    'Sigmoid': 'sigmoid',
    'Log-Sigmoid': 'logsigmoid',
    'Hard-Sigmoid': 'hardsigmoid',
    'Threshold': 'threshold',
    'Soft-Plus': 'softplus',
    'Soft-Shrink': 'softshrink',
    'Soft-Sign': 'softsign'
}

# Training
OPTIMIZERS = {
    'Adaptive Moment Estimation (Adam)': 'adam', 
    'Adam with Infinity Norm (Adamax)': 'adamax', 
    'Adam with Weight Decay (AdamW)': 'adamw', 
    'Rectified Adam (RAdam)': 'radam', 
    'Nesterov-accelerated Adam (NAdam)': 'nadam', 
    'Sparse-Adam (SAdam)': 'sadam', 
    'Adaptive Delta Method (AdaDelta)': 'adadelta', 
    'Adaptive Gradient Algorithm (AdaGrad)': 'adagrad', 
    'Root Mean Square Propagation (RMSProp)': 'rmsprop', 
    'Resilient Backpropagation (RProp)': 'rprop', 
    'Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS)': 'lbfgs', 
    'Averaged SGD (ASGD)': 'asgd', 
    'Stochastic Gradient Descent (SGD)': 'sgd'
}

LR_SCHEDULERS = {
    'Lambda LR': 'lambda',
    'Exponential LR': 'exp',
    'Multiplicative LR': 'mult',
    'Step LR': 'step',
    'Constant LR': 'const',
    'Polynomial LR': 'poly',
    'Multi-Step LR': 'multistep',
    'Cosine Annealing LR': 'cosan',
    'Linear LR': 'linear',
    'Cosine Annealing LR with Warm Restarts': 'cosanwr',
    'Reduce LR on Plateau': 'plateau'
}

BASELINES = ['Rollout', 'Critic', 'Exponential']

# Meta-Reinforcement Learning
MRL_METHODS = {
    '': None,
    'Temporal Difference Learning (TDL)': 'tdl', 
    'Reward Weight Adjustment (RWA)': 'rwa', 
    'Contextual Bandits (CB)': 'cb', 
    'Multi-Objective Reinforcement Learning (MORL)': 'morl'
}

CB_EXPLORATION_METHODS = {
    'Upper Confidence Bound (UCB)': 'ucb', 
    'Thompson Sampling': 'thompson_sampling', 
    'Epsilon-Greedy': 'epsilon_greedy'
}

RWA_MODELS = {'Recurrent Neural Network (RNN)': 'rnn'}

RWA_OPTIMIZERS = {
    'Adam with Infinity Norm (Adamax)': 'adamax', 
    'Adaptive Moment Estimation (Adam)': 'adam', 
    'Adam with Weight Decay (AdamW)': 'adamw', 
    'Rectified Adam (RAdam)': 'radam', 
    'Nesterov-accelerated Adam (NAdam)': 'nadam', 
    'Root Mean Square Propagation (RMSProp)': 'rmsprop'
}

# Hyper-Parameter Optimization
HOP_METRICS = {
    'Training Loss': 'loss', 
    'Validation Loss': 'val_loss', 
    'Training Mean Reward': 'mean_reward', 
    'Mean Absolute Error (MAE)': 'mae', 
    'Mean Squared Error (MSE)': 'mse', 
    'Root MSE (RMSE)': 'rmse', 
    'Episode Mean Reward': 'episode_reward_mean', 
    'Efficiency (Kilogram per Kilometer)': 'kg/km', 
    'Number of Overflows': 'overflows', 
    'Efficiency and Overflows': 'both'
}

HOP_METHODS = {
    '': None,
    'Distributed Evolutionary Algorithm (DEA)': 'dea', 
    'Bayesian Optimization (BO)': 'bo', 
    'Hyperband Optimization (HBO)': 'hbo', 
    'Random Search (RS)': 'rs', 
    'Grid Search (GS)': 'gs', 
    'Differential Evolutionary Hyperband Optimization (DEHBO)': 'dehbo', 
    'Population-Based Bandits Algorithm (PBBA)': 'pbba', 
    'Bayesian Generational Population-Based Training (BGPBT)': 'bgpbt'
}

# Logging
WB_MODES = ['offline', 'online', 'disabled']

# File System Settings
OPERATION_MAP = {
    '': '',
    'Set value (=)': '=',
    'Compare equality (==)': '==',
    'Compare inequality (!=)': '!=',
    'Add value (+)': '+',
    'Subtract value (-)': '-',
    'Multiply by value (*)': '*',
    'Divide by value (/)': '/',
    'Power of value (**)': '**',
    'Floor division by value (//)': '//',
    'Modulo operation (%)': '%',
    'Shift left (<<)': '<<',
    'Check if smaller (<)': '<',
    'Check if smaller or equal (<=)': '<=',
    'Shift right (>>)': '>>',
    'Check if larger (>)': '>',
    'Check if larger or equal (>=)': '>=',
    'Perform bit-wise OR (|)': '|',
    'Perform bit-wise AND (&)': '&',
    'Perform bit-wise XOR (^)': '^',
    'Check if value is new_value (is)': 'is',
    'Check if value is not new_value (is not)': 'isnot',
    'Check if value is in new_value (in)': 'in',
    'Check if value is not in new_value (not in)': 'notin',
    'Matrix multiplication (@)': '@',
    'Division with remainder (divmod - returns tuple)': 'divmod'
}

FUNCTION_MAP = {
    'Mean': 'mean',
    'Standard Deviation (stdev)': 'stdev',
    'Median': 'median',
    'Mode': 'mode',
    'Variance (var)': 'var',
    'Quantiles (quant)': 'quant',
    'Size': 'size',
    'Summmation (sum)': 'sum',
    'Minimum (min)': 'min',
    'Maximum (max)': 'max'
}

# Test suite settings
TEST_MODULES = {
    'parser': 'test_configs_parser.py',
    'train': 'test_train_command.py',
    'mrl': 'test_mrl_train_command.py',
    'hp_optim': 'test_hp_optim_command.py',
    'gen_data': 'test_gen_data_command.py',
    'eval': 'test_eval_command.py',
    'test_sim': 'test_test_command.py',
    'file_system': 'test_file_system_command.py',
    'gui': 'test_gui_command.py',
    'actions': 'test_custom_actions.py',
    'edge_cases': 'test_edge_cases.py',
    'layers': 'test_model_layers.py',
    'scheduler': 'test_lr_scheduler.py',
    'optimizer': 'test_optimizer.py',
    'integration': 'test_integration.py'
}

# Define the metrics for the daily chart tab
TARGET_METRICS = ['overflows', 'kg', 'ncol', 'km', 'kg/km', 'profit']
HEATMAP_METRICS = ['bin_state_c', 'bin_state_collected', 'bins_state_c_after']