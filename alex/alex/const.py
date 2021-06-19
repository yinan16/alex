import pkg_resources
import os


COMPONENT_BASE_PATH = pkg_resources.resource_filename("alex", "./components")
CACHE_BASE_PATH = pkg_resources.resource_filename("alex", "../cache")
EXAMPLES_PATH = pkg_resources.resource_filename("alex", "../examples")
CHECKPOINT_BASE_PATH = pkg_resources.resource_filename("alex", "..checkpoints/")
ENGINE_PATH = pkg_resources.resource_filename("alex", "./engine")

ALEX_CACHE_BASE_PATH = ".alex.cache/"
os.makedirs(ALEX_CACHE_BASE_PATH, exist_ok=True)

DEFAULT_DTYPE = "float32"
# ---------------------------- Trainable parameters ------------------------- #
CONSTRUCTORS = {"params": {"tf": ["tf.Variable",
                                  {"initial_value": "initializer",
                                   "trainable": "is_trainable",
                                   "caching_device": "None",
                                   "name": "name",
                                   "variable_def": "None",
                                   "dtype": "tf_dtypes[dtype]",
                                   "import_scope": "None",
                                   "constraint": "None",
                                   "synchronization": "tf.VariableSynchronization.AUTO",
                                   "shape": "None"}],
                           "keras": ["", []],
                           "pytorch": ["torch.nn.parameter.Parameter",
                                       {"data": "initializer",
                                        "requires_grad": "is_trainable"}],
                           "alex": ["params",
                                    {"initializer": False,
                                     "name": True,
                                     "dtype": True,
                                     "is_trainable": True}]}}


PARAMS = {"conv": {"filters": {"derivative": True,
                               "ingredient": "conv",
                               "shape": {"pytorch": ["[n_filters, input_shape, kernel_size_h, kernel_size_w]",
                                                     {}],
                                         "tf": ["(kernel_size_h, kernel_size_w, input_shape, n_filters)",
                                                {}],
                                         "alex": ["conv_filters_shape",
                                                  {"n_filters": True,
                                                   "input_shape": False,
                                                   "kernel_size_h": True,
                                                   "kernel_size_w": True}]}}},
          "dense": {"weights": {"derivative": True,
                                "ingredient": "dense",
                                "shape":
                                {"pytorch": ["[n_units, input_shape]",
                                             {}],
                                 "tf": ["[input_shape, n_units]",
                                        {}],
                                 "alex": ["dense_weights_shape",
                                          {"n_units": True,
                                           "input_shape": False}]}},
                    "bias": {"derivative": True,
                             "ingredient": "dense",
                             "shape":
                             {"alex": ["dense_bias_shape",
                                          {"dim": True}],
                              "pytorch": ["[dim, ]",
                                          {}],
                              "tf": ["[dim, ]",
                                     {}]}}},
          "batch_normalize": {"mean": {"derivative": False,
                                       "ingredient": "batch_normalize",
                                       "shape": {"pytorch":
                                                 ["[input_shape, ]",
                                                  {}],
                                                 "tf":
                                                 ["[input_shape, ]",
                                                  {}],
                                                 "alex":
                                                 ["batch_normalize_mean_shape",
                                                  {"input_shape": False}]}},
                              "variance": {"derivative": False,
                                           "ingredient": "batch_normalize",
                                           "shape": {"pytorch":
                                                     ["[input_shape, ]",
                                                      {}],
                                                     "tf":
                                                     ["[input_shape, ]",
                                                      {}],
                                                     "alex":
                                                     ["batch_normalize_variance_shape",
                                                      {"input_shape": False}]}},
                              "offset": {"derivative": True,
                                         "ingredient": "batch_normalize",
                                         "shape": {"pytorch":
                                                   ["[input_shape, ]",
                                                    {}],
                                                   "tf":
                                                   ["[input_shape, ]",
                                                    {}],
                                                   "alex":
                                                   ["batch_normalize_offset_shape",
                                                    {"input_shape": False}]}},
                              "scale": {"derivative": True,
                                        "ingredient": "batch_normalize",
                                        "shape": {"pytorch":
                                                  ["[input_shape, ]",
                                                   {}],
                                                  "tf":
                                                  ["[input_shape, ]",
                                                   {}],
                                                  "alex":
                                                  ["batch_normalize_scale_shape",
                                                   {"input_shape": False}]}},
          }
}

ALL_PARAMS = dict()
ALL_TRAINABLE_PARAMS = dict()
for ingredient in PARAMS:
    for param in PARAMS[ingredient]:
        ALL_PARAMS[param] = PARAMS[ingredient][param]
        if PARAMS[ingredient][param]["derivative"]:
            ALL_TRAINABLE_PARAMS[param] = PARAMS[ingredient][param]

ALL_PARAMS_LIST = list(ALL_PARAMS.keys())
ALL_TRAINABLE_PARAMS_LIST = list(ALL_TRAINABLE_PARAMS.keys())

INITIALIZERS_NO_HYPE = {"zeros_initializer": {"keras": ["",
                                                        []],
                                              "tf": ["tf.zeros_initializer",
                                                     {"shape": "shape",
                                                     "name": "name"}],
                                              "pytorch": ["torch.nn.init.zeros_",
                                                          {"tensor": "torch.empty(*shape)"}],
                                              "alex": ["zeros_initializer",
                                                       {"dtype": True,
                                                        "shape": False,
                                                        "name": True}]},
                        "ones_initializer": {"keras": ["",
                                                       []],
                                             "tf": ["tf.ones_initializer",
                                                    {"shape": "shape",
                                                     "name": "name"}],
                                             "pytorch": ["torch.nn.init.ones_",
                                                         {"tensor": "torch.empty(*shape)"}],
                                             "alex": ["ones_initializer",
                                                      {"dtype": True,
                                                       "shape": False,
                                                       "name": True}]}}

INITIALIZERS = {"he_uniform": {"keras": ["keras.initializers.he_uniform", []],
                               "tf": ["tf.keras.initializers.he_uniform(seed=seed)",
                                      {"shape": "shape",
                                       "name": "name"}],
                               "pytorch": ["torch.nn.init.kaiming_uniform",
                                           {"tensor": "torch.empty(*shape)"}],
                               "alex": ["he_uniform",
                                        {"dtype": True,
                                         "seed": True,
                                         "shape": False,
                                         "name": True}]},
                "xavier_uniform": {"keras": ["keras.initializers.glorot_uniform",
                                             {}],
                                   "tf": ["tf.keras.initializers.glorot_uniform(seed=seed)",
                                          []],
                                   "pytorch": ["torch.nn.init.xavier_uniform_",
                                               {"tensor": "torch.empty(*shape)"}],
                                   "alex": ["xavier_uniform",
                                            {"dtype": True,
                                             "seed": True,
                                             "shape": False,
                                             "name": True}]}}

ALL_INITIALIZERS = {**INITIALIZERS_NO_HYPE, **INITIALIZERS}

COMPONENTS = {"conv": {"tf": ["tf.nn.conv2d",
                              {"input": "inputs",
                               "filters": "filters",
                               "strides": "strides",
                               "padding": "padding",
                               "data_format": "'NHWC'",
                               "dilations": "dilation",
                               "use_bias": "False",
                               "name": "name"}],
                       "keras": ["keras.layers.Conv2D", {}],
                       "pytorch": ["torch.nn.functional.conv2d",
                                   {"input": "inputs",
                                    "weight": "filters",
                                    "bias": "None",
                                    "stride": "strides",
                                    "padding": "padding",
                                    "dilation": "dilation",
                                    "groups": "1"}],
                       "alex": ["conv",
                                {"filters": False,
                                 "padding": True,
                                 "strides": True,
                                 "dilation": True,
                                 "inputs": False,
                                 "dtype": True,
                                 "name": True,
                                 "training": False}]},
              "dense": {"tf": ["tf.add(x=tf.matmul(a=inputs, b=weights), y=bias, name=name)",
                               {}], # FIXME
                        "keras": ["keras.layers.Dense", []],
                        "pytorch": ["torch.nn.functional.linear",
                                    {"weight": "weights",
                                     "bias": "bias",
                                     "input": "inputs"}],
                        "alex": ["dense",
                                 {"inputs": False,
                                  "weights": False,
                                  "bias": False,
                                  "dtype": True,
                                  "name": True}]},
              "batch_normalize": {"tf": ["tf.nn.batch_normalization",
                                         {"x": "inputs",
                                          "mean": "mean",
                                          "variance": "variance",
                                          "offset": "offset",
                                          "scale": "scale",
                                          "variance_epsilon": "epsilon",
                                          "name": "name"}],
                                  "pytorch": ["torch.nn.functional.batch_norm",
                                              {"input": "inputs",
                                               "running_mean": "mean",
                                               "running_var": "variance",
                                               "weight": "scale",
                                               "bias": "offset",
                                               "training": "training",
                                               "momentum": "momentum", "eps": "epsilon"}],
                                  "alex": ["batch_normalize",
                                           {"mean": False,
                                            "variance": False,
                                            "offset": False,
                                            "scale": False,
                                            "momentum": True,
                                            "epsilon": True,
                                            "inputs": False,
                                            "name": True,
                                            "dtype": True,
                                            "training": False}]}
}

# 'deconv': {"tf": "tf.nn.conv2d_transpose"},

# ---------------------------- Deterministic layers -------------------------- #
DET_COMPONENTS_HYPE = {"dropout": {"tf": ["tf.nn.dropout",
                                          {"x": "inputs",
                                           "rate": "dropout_rate",
                                           "noise_shape": "None",
                                           "seed": "None",
                                           "name": "name"}],
                                   "keras": ["keras.layers.Dropout", []],
                                   "pytorch": ["torch.nn.functional.dropout",
                                               {"input": "inputs",
                                                "p": "dropout_rate",
                                                "training": "training",
                                                "inplace": "False"}],
                                   "alex": ["dropout",
                                            {"dropout_rate": True,
                                             "inputs": False,
                                             "name": True,
                                             "dtype": True,
                                             "training": False}]},
                       "max_pool2d": {"tf": ["tf.nn.max_pool",
                                             {"input": "inputs",
                                              "ksize": "window_shape",
                                              "strides": "strides",
                                              "padding": "padding",
                                              "data_format": "'NHWC'",
                                              "name": "name"}],
                                      "pytorch": ["torch.nn.functional.max_pool2d",
                                                  {"input": "inputs",
                                                   "kernel_size": "window_shape",
                                                   "stride": "strides",
                                                   "padding": "padding"}],
                                      "alex": ["max_pool2d",
                                               {"window_shape": True,
                                                "strides": True,
                                                "padding": True,
                                                "dilation": True,
                                                "inputs": False,
                                                "dtype": True,
                                                "name": True}]}}

DET_COMPONENTS_NO_HYPE = {"relu": {"tf": ["tf.nn.relu",
                                          {"name": "name",
                                           "features": "inputs"}],
                                   "keras": ["keras.layers.ReLU", []],
                                   "pytorch": ["torch.nn.functional.relu",
                                               {"input": "inputs",
                                                "inplace": "False"}],
                                   "alex": ["relu",
                                            {"inputs": False,
                                             "dtype": True,
                                             "name": True}]},
                          "softmax": {"tf": ["tf.nn.softmax",
                                             {}],
                                      "pytorch": ["torch.nn.functional.softmax",
                                                  {"input": "inputs",
                                                   "dim": "None"}],
                                      "alex": ["softmax",
                                               {"inputs": False,
                                                "dtype": True,
                                                "name": True}]},
                          "sigmoid": {"tf": ["tf.math.sigmoid",
                                             {}],
                                      "pytorch": ["torch.sigmoid",
                                                  {"input": "inputs"}],
                                      "alex": ["sigmoid",
                                               {"inputs": False,
                                                "dtype": True,
                                                "name": True}]},
                          "flatten": {"tf": ["tf.reshape",
                                             {"tensor": "inputs",
                                              "shape": "(-1, shape.prod())",
                                              "name": "name"}],
                                      "keras": ["keras.layers.Flatten", []],
                                      "pytorch": ["torch.flatten",
                                                  {"input": "inputs",
                                                   "start_dim": "1",
                                                   "end_dim": "-1"}],
                                      "alex": ["flatten",
                                               {"inputs": False,
                                                "shape": False,
                                                "dtype": True,
                                                "name": True}]},
                          "add": {"tf": ["tf.math.add",
                                         {"x": "inputs[0]",
                                          "y": "inputs[1]",
                                          "name": "name"}],
                                  "pytorch": ["torch.add",
                                              {"input": "inputs[0]",
                                               "other": "inputs[1]"}],
                                  "alex": ["add",
                                           {"inputs": False,
                                            "dtype": True,
                                            "name": True}]},
                          "zeros": {"tf": ["tf.zeros",
                                           {}],
                                    "pytorch": ["torch.zeros",
                                                {"input_shape": "shape"}],
                                    "alex": ["zeros",
                                             {"shape": False,
                                              "name": True}]}}
# TODO: concatenate, softmax
DL_LAYERS = {**COMPONENTS,
             **DET_COMPONENTS_HYPE,
             **DET_COMPONENTS_NO_HYPE}

INFERENCE = {} # inference function
LOSSES = {"cross_entropy": {"tf": ["tf.nn.softmax_cross_entropy_with_logits",
                                   {"labels": "inputs[0]",
                                    "logits": "inputs[1]",
                                    "axis": "-1",
                                    "name": "name"}],
                            "pytorch": ["torch.nn.functional.cross_entropy",
                                        {"weight": "None",
                                         "ignore_index": "-100",
                                         "reduction": "'mean'",
                                         "target": "inputs[0]",
                                         "input": "inputs[1]"}],
                            "alex": ["cross_entropy",
                                     {"name": True,
                                      "dtype": True,
                                      "training": False}]}}
# TODO: bitwise, l2, mean_square
REGULARIZERS = {"regularizer_l2": {"tf": ["coeff*sum(list(map(lambda x: tf.nn.l2_loss(x=trainable_params[x], name=name), inputs)))",
                                          {}],
                                   "pytorch": ["coeff*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), inputs)))",
                                               {}],
                                   "keras": ["tf.keras.regularizers.L2", []],
                                   "alex": ["regularizer_l2",
                                            {"coeff": True,
                                             "name": True,
                                             "training": False,
                                             "dtype": True,
                                             "trainable_params": False,
                                             "inputs": False}]}}

OPTIMIZERS = {"adam": {"tf": ["tf.optimizers.Adam",
                              {"learning_rate": "learning_rate",
                               "beta_1": "beta1",
                               "beta_2": "beta2",
                               "epsilon": "epsilon",
                               "name": "name"}],
                       "pytorch": ["torch.optim.Adam",
                                   {"params": "trainable_params",
                                    "lr": "learning_rate",
                                    "betas": "(beta1, beta2)",
                                    "eps": "epsilon"
                                   }],
                       "alex": ["adam",
                                {"learning_rate": True,
                                 "beta1": True,
                                 "beta2": True,
                                 "epsilon": True,
                                 "inputs": False,
                                 "name": True,
                                 "training": False,
                                 "dtype": True,
                                 "trainable_params": False,
                                 "initial_learning_rate": True}]}
}
# TODO: gradient_descent, momentum_optimizer
LOSS_BLOCK = {**LOSSES, **REGULARIZERS}
INGREDIENT_TYPES = {**DL_LAYERS, **LOSSES, **OPTIMIZERS, **INFERENCE}

INPUTS = ("data", "label")


LEARNING_RATE_DECAY = {"exponential_decay": {"tf": ["tf.keras.optimizers.schedules.ExponentialDecay",
                                                    {"initial_learning_rate": "initial_learning_rate",
                                                     "decay_steps": "decay_steps",
                                                     "decay_rate": "decay_rate",
                                                     "staircase": "staircase"}],
                                             "pytorch": ["torch.optim.lr_scheduler.ExponentialLR",
                                                         {"optimizer": "optimizer",
                                                          "gamma": "decay_rate",
                                                          "last_epoch": "-1",
                                                          "verbose": "False"
                                                         }],
                                             "alex": ["exponential_decay",
                                                      {"initial_learning_rate": True,
                                                       "decay_steps": True,
                                                       "decay_rate": True,
                                                       "staircase": True
                                                      }]}} # FIXME
SCHEDULER_BLOCK = {**LEARNING_RATE_DECAY}

OPTIMIZER_BLOCK = {**OPTIMIZERS}

TENSOR_SHAPE = {"input_shape": {"tf": ["inputs.get_shape().as_list()", []]}}

AS_TENSOR = {"as_tensor": {"tf": ["tf.convert_to_tensor",
                                  {"value": "np.asarray(inputs)",
                                   "dtype": "tf_dtypes['dtype']",
                                   "dtype_hint": "None"}],
                           "pytorch": ["torch.as_tensor",
                                       {"data": "np.asarray(inputs)",
                                        "dtype": "torch_types[dtype]",
                                        "device": "device"}],
                           "alex": ["as_tensor", {"inputs": False,
                                                  "dtype": True,
                                                  "device": False,
                                                  "name": True}]}}
COMPONENT_BLOCK = {**DL_LAYERS, **AS_TENSOR}

UTILITY = {**TENSOR_SHAPE, **LEARNING_RATE_DECAY, **AS_TENSOR}

ALL_COMPONENTS = {**COMPONENTS,
                  **ALL_INITIALIZERS,
                  **DET_COMPONENTS_HYPE,
                  **DET_COMPONENTS_NO_HYPE,
                  **INFERENCE,
                  **LOSSES,
                  **REGULARIZERS,
                  **OPTIMIZERS,}
                  # **UTILITY}

ALL_RECIPES = {"root",
               "adense",
               "resnet_16",
               "resnet_32", "resnet_32_short_cut",
               "resnet_64", "resnet_64_short_cut",
               "resnet_128", "resnet_128_short_cut",
               "resnet_256", "resnet_256_short_cut",
               "resnet_512", "resnet_512_short_cut"}

TYPES = {0: "DETERMINISTIC_COMPONENTS_WITHOUT_HYPERPARAMETER",
         1: "DETERMINISTIC_COMPONENTS_WITH_HYPERPARAMETER",
         2: "TRAINABLE_COMPONENTS"}

# ------------------------- User defined functions --------------------------- #

USER_FNS = {"lr_scheduler": "engine.ns_user.learning_rate"}


FNS = [*list(ALL_COMPONENTS.keys()),
       *list(USER_FNS.keys()),
       *list(SCHEDULER_BLOCK.keys()),
       *ALL_TRAINABLE_PARAMS, "shape"]

# -------------------------------- String consts ----------------------------- #
# Network build related
COMPONENT_LIST = 'component_list'
LOSS_LIST = 'loss_list'

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
INPUT_COMPONENT = 'input_component'
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
