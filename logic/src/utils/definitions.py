import os
import statistics

from pathlib import Path


# Paths
path = Path(os.getcwd())
parts = path.parts
ROOT_DIR = Path(*parts[:parts.index('WSmart-Route') + 1])
ICON_FILE = os.path.join(ROOT_DIR, 'assets', 'images', "logo-wsmartroute-white.png")

# Multi-core processing settings
CORE_LOCK_WAIT_TIME=10
LOCK_TIMEOUT=CORE_LOCK_WAIT_TIME

def update_lock_wait_time(num_cpu_cores=None):
    """
    Updates the global LOCK_TIMEOUT based on the number of CPU cores.
    
    Returns:
        The new (or default) value of LOCK_TIMEOUT.
    """
    global LOCK_TIMEOUT
    global CORE_LOCK_WAIT_TIME
    if num_cpu_cores is None:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME
    else:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT

# Waste management information
MAP_DEPOTS = {
    'mixrmbac': 'CTEASO', # Rio Maior, Bombarral, Azambuja, Cadaval
    'riomaior': 'CTEASO',
    'figueiradafoz': 'CITVRSU'
}

WASTE_TYPES = {
    'glass': 'Embalagens de Vidro',
    'plastic': 'Mistura de embalagens',
    'paper': 'Embalagens de papel e cartão'
}

# Distance matrix
EARTH_RADIUS = 6371
EARTH_WMP_RADIUS = 6378137

# WSmart+ route simulation settings
PBAR_WAIT_TIME=0.1

METRICS = ['overflows', 'kg', 'ncol', 'kg_lost', 'km', 'kg/km', 'cost', 'profit']
SIM_METRICS = METRICS + ['days', 'time']
DAY_METRICS = ['day'] + METRICS + ['tour']
LOSS_KEYS = ['nll', 'reinforce_loss', 'baseline_loss']
TQDM_COLOURS = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'white'] #, 'black']

# Plotting
MARKERS = ['P', 's', '^', '8', '*']
PLOT_COLOURS = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'black'] #, 'white']
LINESTYLES = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']

# Problem definition
MAX_WASTE = 1.
MAX_LENGTHS = {
    20: 2,
    50: 3,
    100: 4,
    150: 5,
    225: 6,
    317: 7
}
VEHICLE_CAPACITY = 100.

# Model configurations
SUB_NET_ENCS = ['tgc']
PRED_ENC_MODELS = ['tam']
ENC_DEC_MODELS = ['ddam']

# Hyper-Parameter Optimization
HOP_KEYS = (
    'hop_method', 'hop_range', 'hop_epochs', 'metric',                              # Hyper-Parameter Optimization (HPO)
    'n_trials', 'timeout', 'n_startup_trials', 'n_warmup_steps', 'interval_steps',  # Bayesian Optimization (BO)
    'eta', 'indpb', 'tournsize', 'cxpb', 'mutpb', 'n_pop', 'n_gen',                 # Distributed Evolutionary Algorithm (DEA)
    'fevals',                                                                       # Differential Evolutionary Hyperband Optimization (DEHBO)
    'cpu_cores', 'verbose', 'train_best', 'local_mode', 'num_samples',              # Ray Tune framework
    'max_tres', 'reduction_factor',                                                 # Hyperband Optimization (HBO) - Ray Tune
    'max_failures',                                                                 # Random Search (RS) - Ray Tune
    'grid', 'max_conc',                                                             # Grid Search (GS) - Ray Tune
)

# File system settings
CONFIRM_TIMEOUT=30

FS_COMMANDS = ['create', 'read', 'update', 'delete', 'cryptography']

OPERATION_MAP = {
    '=': lambda x, y: y,                    # Replace with right value
    '==': lambda x, y: x == y,              # Compare equality of both values
    '!=': lambda x, y: x != y,              # Compare inequality of both values
    '+': lambda x, y: x + y,                # Add value
    '+=': lambda x, y: x + y,               # Same as '+'
    '-': lambda x, y: x - y,                # Subtract value
    '-=': lambda x, y: x - y,               # Same as '-'
    '*': lambda x, y: x * y,                # Multiply by value
    '*=': lambda x, y: x * y,               # Same as '*'
    '/': lambda x, y: x / y,                # Divide by value
    '/=': lambda x, y: x / y,               # Same as '/'
    '**': lambda x, y: x ** y,              # Power of value
    '**=': lambda x, y: x ** y,             # Same as '**'
    '//': lambda x, y: x // y,              # Floor division by value
    '//=': lambda x, y: x // y,             # Same as '//'
    '%': lambda x, y: x % y,                # Modulo operation
    '%=': lambda x, y: x % y,               # Same as '%'
    '': lambda x, y: x,                     # Dont perform any change
    '<<': lambda x, y: x << y,              # Perform left shit
    '<<=': lambda x, y: x << y,             # Same as '<<'
    '<': lambda x, y: x < y,                # Check if left value is smaller
    '<=': lambda x, y: x <= y,              # Check if left value is smaller or equal
    '>>': lambda x, y: x >> y,              # Perform right shift
    '>>=': lambda x, y: x >> y,             # Same as '>>'
    '>': lambda x, y: x > y,                # Check if left value is larger
    '>=': lambda x, y: x >= y,              # Check if left value is larger or equal
    '|': lambda x, y: x | y,                # Perform bit-wise OR operation
    '|=': lambda x, y: x | y,               # Same as '|'
    '&': lambda x, y: x & y,                # Perform bit-wise AND operation
    '&=': lambda x, y: x & y,               # Same as '&'
    '^': lambda x, y: x ^ y,                # Perform bit-wise XOR operation
    '^=': lambda x, y: x ^ y,               # Same as '^'
    'is': lambda x, y: x is y,              # Compares identity
    'isnot': lambda x, y: x is not y,       # Compares negation of identity
    'in': lambda x, y: x in y,              # Checks membership
    'notin': lambda x, y: x not in y,       # Checks negation of membership
    '@': lambda x, y: x @ y,                # Perform matrix multiplication
    '@=': lambda x, y: x @ y,               # Same as '@'
    'divmod': lambda x, y: divmod(x, y),    # Division with remainder (return tuple)
}

STATS_FUNCTION_MAP = {
    'mean': statistics.mean,
    'stdev': statistics.stdev,
    'median': statistics.median,
    'mode': statistics.mode,
    'var': statistics.variance,
    'quant': statistics.quantiles,
    'size': len,
    'sum': sum,
    'min': min,
    'max': max
}

# GUI settings
CTRL_C_TIMEOUT = 2.0

APP_STYLES = ['fusion', 'windows', 'windowsxp', 'macintosh']

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