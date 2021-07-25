import pkg_resources
import os


COMPONENT_BASE_PATH = pkg_resources.resource_filename("alex", "./components")
CACHE_BASE_PATH = pkg_resources.resource_filename("alex", "../cache")
EXAMPLES_PATH = pkg_resources.resource_filename("alex", "../examples")
CHECKPOINT_BASE_PATH = pkg_resources.resource_filename("alex", "..checkpoints/")
ENGINE_PATH = pkg_resources.resource_filename("alex", "./engine")

ALEX_CACHE_BASE_PATH = ".alex.cache/"
os.makedirs(ALEX_CACHE_BASE_PATH, exist_ok=True)

# -------------------------------- String consts ----------------------------- #
# Network build related
YAML = ".yml"
COMPONENT_NAMES = 'component_types'
NAME = 'name'
DEFINITION = 'definition'
REPEAT = 'repeat'
SHARE = "share"
KEEP_TRACK = 'keep_track'
RUNTIME = "runtime"
RUNTIMES = "runtimes"
DATA = "data"
LABEL = "label"
# General config
LOGS = 'logs'
VAR = 'var'
VAR_NAME = "var_name"
STATS = "stats"
COMPONENT = 'component'
LOSS = 'loss'
SOLVER = 'optimizers'
TYPE = 'type'
INPUTS = 'inputs'
PARAMS_NETWORK = 'params_network'
PARAMS_TRAINING = "params_training"
PARAMS_LOSSES = 'params_losses'
PARAMS_SOLVERS = 'params_optimizers'
HYPERPARAMS = 'hyperparams'
IS_TRAINING = 'is_training'
LEARNING_RATE = 'learning_rate'
CONFIG = "config"
STAT = "stat"
VALUE = "value"
TENSOR = "tensor"
CHECKPOINT_DIR = "checkpoint_dir"
CHECKPOINT_EXT = ".json"
INDEX = "index"
EVAL = "eval"
INFERENCE = 'inference'
###############################################
# Graph related:
META = "meta"
SUBGRAPH = "subgraph"
GRAPH = "graph"
ROOT = "root"
SCOPE = "scope"
LEVEL = "level"
SEQUENTIAL = "sequential"
###############################################
# Component related
# Component hyperparams
### general
ACTIVATION_FUNCTION = 'activation_function'
INITIALIZATION = 'initialization'
INITIALIZER = "initializer"
RANDOM = 'random'
OUTPUT_SHAPE = 'output_shape'
SHAPE = "shape"
### conv2d
STRIDES = 'strides'
FILTERS = 'filters'
N_FILTERS = 'n_filters'
KERNEL_SIZE = 'kernel_size'

### dense
N_UNITS = 'n_units'

### batch_norm
VARIANCE_EPSILON = "variance_epsilon"
DECAY = "decay"
# pool:
WINDOW_SHAPE = 'window_shape'
POOLING_TYPE = 'pooling_type'
PADDING = 'padding'

### training
WEIGHT_DECAY = "weight_decay"
DROPOUT_RATE = 'dropout_rate'

###############################################
# Data related
DATA_NAME = 'data_name'
TRAINING_DATA = 'training_data'
TESTING_DATA = 'testing_data'
INPUT_DATA = 'input_data'
LABELS = 'labels'
MEAN = 'mean'
STD = 'std'
DTYPE = "dtype"
# Data hyperparams
BATCH_SIZE = 'batch_size'
TEST_SIZE = 'test_size'

################################################
## 20200709
RPT_IDX = "__rpt_idx__" # repeat index

COMPONENT_INFO_KEYS = [VALUE, HYPERPARAMS, STAT, NAME, VAR]

#################################################
# alex namespace
# arguments
ALEX_ARG_NAME = "name"
ALEX_ARG_INPUTS = "inputs"
ALEX_ARG_LABELS = "labels"
ALEX_ARG_INITIAL_VALUE = "initial_value"
ALEX_ARG_INPUT_SHAPE = "input_shape"
ALEX_ARG_SHAPE = "shape"
ALEX_ARG_TRAINABLE_PARAMS = "trainable_params"
ALEX_ARG_TRAINABLE_PARAMS_NAME = "trainable_params_name"
