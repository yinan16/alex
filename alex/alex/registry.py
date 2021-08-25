# ----------------------------------------------------------------------
# Created: sön jul 25 13:12:20 2021 (+0200)
# Last-Updated:
# Filename: registry.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

DEFAULT_DTYPE = "float32"
# -------------------------------------------------------------------------- #
# Definitions
# -------------------------------------------------------------------------- #
# Utilities

TENSOR_SHAPE = {"input_shape": {"tf": ["inputs.get_shape().as_list()", []]}}

AS_TENSOR = {"as_tensor": {"tf": ["tf.convert_to_tensor",
                                  {"value": "np.asarray(inputs)",
                                   "dtype": "tf_dtypes[dtype]",
                                   "dtype_hint": "None"}],
                           "pytorch": ["torch.as_tensor",
                                       {"data": "np.asarray(inputs)",
                                        "dtype": "torch_types[dtype]",
                                        "device": "device"}],
                           "alex": ["as_tensor", {"inputs": False,
                                                  "dtype": True,
                                                  "device": False,
                                                  "name": True}]}}


UTILITIES = {**TENSOR_SHAPE, **AS_TENSOR}

# -------------------------------- Parameters -------------------------------- #
PARAM_CONSTRUCTORS = {"params": {"tf": ["tf.Variable",
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


# TODO: as it is now, the param names need to be globally unique
# "linear2d": {"lr_weights": {"derivative": True,
#                                       "ingredient": "linear2d",
#                                       "shape": {"pytorch": ["[kernel_size_h*kernel_size_w*input_size, ]", {}]}}}, # FIXME: WIP

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
ALL_OTHER_PARAMS = dict()
for ingredient in PARAMS:
    for param in PARAMS[ingredient]:
        ALL_PARAMS[param] = PARAMS[ingredient][param]
        if PARAMS[ingredient][param]["derivative"]:
            ALL_TRAINABLE_PARAMS[param] = PARAMS[ingredient][param]
        else:
            ALL_OTHER_PARAMS[param] = PARAMS[ingredient][param]


INITIALIZERS_WITHOUT_HYPE = {"zeros_initializer": {"keras": ["",
                                                             []],
                                                   "tf": ["tf.zeros_initializer()",
                                                          {"shape": "shape"}],
                                                   "pytorch": ["torch.nn.init.zeros_",
                                                               {"tensor": "torch.empty(*shape)"}],
                                                   "alex": ["zeros_initializer",
                                                            {"dtype": True,
                                                             "shape": False}]},
                             "ones_initializer": {"keras": ["",
                                                            []],
                                                  "tf": ["tf.ones_initializer()",
                                                         {"shape": "shape"}],
                                                  "pytorch": ["torch.nn.init.ones_",
                                                              {"tensor": "torch.empty(*shape)"}],
                                                  "alex": ["ones_initializer",
                                                           {"dtype": True,
                                                            "shape": False}]}}

INITIALIZERS_WITH_HYPE = {"he_uniform": {"keras": ["keras.initializers.he_uniform", []],
                                         "tf": ["tf.keras.initializers.he_uniform(seed=seed)",
                                                {"shape": "shape"}],
                                         "pytorch": ["torch.nn.init.kaiming_uniform",
                                                     {"tensor": "torch.empty(*shape)"}],
                                         "alex": ["he_uniform",
                                                  {"dtype": True,
                                                   "seed": True,
                                                   "shape": False}]},
                          "xavier_uniform": {"keras": ["keras.initializers.glorot_uniform",
                                                       {}],
                                             "tf": ["tf.keras.initializers.glorot_uniform(seed=seed)",
                                                    {"shape": "shape"}],
                                             "pytorch": ["torch.nn.init.xavier_uniform_",
                                                         {"tensor": "torch.empty(*shape)"}],
                                             "alex": ["xavier_uniform",
                                                      {"dtype": True,
                                                       "seed": True,
                                                       "shape": False}]}}

ALL_INITIALIZERS = {**INITIALIZERS_WITHOUT_HYPE, **INITIALIZERS_WITH_HYPE}

PARAM_BLOCK = {**PARAM_CONSTRUCTORS, **ALL_INITIALIZERS, **ALL_PARAMS}
# ----------------------------- Model ---------------------------------------- #
## ---------------------------- Stateful layer ------------------------------ ##
STATEFUL_INGREDIENTS = {"conv": {"tf": ["tf.nn.conv2d",
                                       {"input": "inputs",
                                        "filters": "filters",
                                        "strides": "strides",
                                        "padding": "padding",
                                        "data_format": "'NHWC'",
                                        "dilations": "dilation",
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

## ---------------------------- Stateless layers -------------------------- ##
STATELESS_INGREDIENTS_WITH_HYPE = {"dropout": {"tf": ["tf.nn.dropout",
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

STATELESS_INGREDIENTS_WITHOUT_HYPE = {"relu": {"tf": ["tf.nn.relu",
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
                                                        {"logits": "inputs",
                                                         "name": "name"}],
                                                 "pytorch": ["torch.nn.functional.softmax",
                                                             {"input": "inputs",
                                                              "dim": "None"}],
                                                 "alex": ["softmax",
                                                          {"inputs": False,
                                                           "dtype": True,
                                                           "name": True}]},
                                     "sigmoid": {"tf": ["tf.math.sigmoid",
                                                        {"x": "inputs",
                                                         "name": "name"}],
                                                 "pytorch": ["torch.sigmoid",
                                                             {"input": "inputs"}],
                                                 "alex": ["sigmoid",
                                                          {"inputs": False,
                                                           "dtype": True,
                                                           "name": True}]},
                                     "flatten": {"tf": ["tf.reshape",
                                                        {"tensor": "inputs",
                                                         "shape": "(-1, tf.math.reduce_prod(tf.convert_to_tensor(shape)))",
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
MODEL_INGREDIENTS = {**STATEFUL_INGREDIENTS,
                     **STATELESS_INGREDIENTS_WITH_HYPE,
                     **STATELESS_INGREDIENTS_WITHOUT_HYPE}

MODEL_BLOCK = {**MODEL_INGREDIENTS, **UTILITIES}

# -------------------------------- Loss -------------------------------------- #
CLASSIFICATION_LOSSES = {"cross_entropy": {"tf": ["tf.nn.softmax_cross_entropy_with_logits",
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
                                                     "inputs": False,
                                                     "dtype": True,
                                                     "training": False}]}}
REGRESSION_LOSSES = {"mse": {"pytorch": ["torch.nn.functional.mse_loss",
                              {"input": "inputs[0]",
                               "target": "inputs[1]",
                               "size_average": "None",
                               "reduce": "None",
                               "reduction": "'mean'"}],
                             "alex": ["mse", {"inputs": False}],
                             "tf": ["tf.keras.metrics.mean_squared_error",
                                    {"y_true": "inputs[0]",
                                     "y_pred": "inputs[1]"}]}}
LOSSES = {**CLASSIFICATION_LOSSES, **REGRESSION_LOSSES}

# TODO: bitwise, l2, mean_square
REGULARIZERS = {"regularizer_l2": {"tf": ["coeff*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name=name), inputs)))",
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

LOSS_BLOCK = {**LOSSES, **REGULARIZERS}

# -------------------------------- Optimizer --------------------------------- #
# TODO: gradient_descent, momentum_optimizer
OPTIMIZER_INGREDIENTS = {"adam": {"tf": ["tf.optimizers.Adam",
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
                                            "initial_learning_rate": True}]}}

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
                                                       "staircase": True,
                                                       "optimizer": False
                                                      }]}} # FIXME
SCHEDULER_BLOCK = {**LEARNING_RATE_DECAY}
OPTIMIZER_BLOCK = {**OPTIMIZER_INGREDIENTS}
# ---------------------------------- Inference ----------------------------- #
INFERENCES = {} # inference function

# ------------------------- User defined functions --------------------------- #

USER_FNS = {} #"lr_scheduler": "engine.ns_user.learning_rate"

ALL_FNS = {**STATEFUL_INGREDIENTS,
           **ALL_INITIALIZERS,
           **STATELESS_INGREDIENTS_WITH_HYPE,
           **STATELESS_INGREDIENTS_WITHOUT_HYPE,
           **INFERENCES,
           **LOSSES,
           **REGULARIZERS,
           **OPTIMIZER_INGREDIENTS,
           **UTILITIES,
           **USER_FNS,
           **ALL_PARAMS}

PREDEFINED_RECIPES = {"adense",
                      "test_recipe",
                      "resnet_16",
                      "resnet_32", "resnet_32_short_cut",
                      "resnet_64", "resnet_64_short_cut",
                      "resnet_128", "resnet_128_short_cut",
                      "resnet_256", "resnet_256_short_cut",
                      "resnet_512", "resnet_512_short_cut"}

BLOCKS = {"data_block",
          "model_block",
          "loss_block",
          "optimizer_block"}

DATA_BLOCK = {"data", "label"}


# -------------------------------------------------------------------------- #
# Types (sets)
# -------------------------------------------------------------------------- #
INGREDIENT_TYPES = {*DATA_BLOCK,
                    *MODEL_INGREDIENTS,
                    *LOSSES,
                    *REGULARIZERS,
                    *OPTIMIZER_INGREDIENTS,
                    *INFERENCES}

RECIPE_TYPES = {"root",
                *BLOCKS,
                *PREDEFINED_RECIPES}


PARAM_TYPES = set(ALL_PARAMS.keys())
TRAINABLE_PARAM_TYPES = set(ALL_TRAINABLE_PARAMS.keys())
NONTRAINABLE_PARAM_TYPES = set(ALL_OTHER_PARAMS.keys())


# Simple function (without composition)
TAG_FUNCTIONS = {*ALL_FNS, "shape"}


TYPES = {0: "STATELESS_INGREDIENTS_WITHOUT_HYPE",
         1: "STATELESS_INGREDIENTS_WITH_HYPE",
         2: "STATEFUL_INGREDIENTS"}


def get_trainable_params_list(params):
    trainable_params_list = []
    for tv in params:
        if tv.split('/')[-1] in ALL_TRAINABLE_PARAMS:
            trainable_params_list.append(params[tv])
    return trainable_params_list
# -------------------------------------------------------------------------- #
# Summary
# -------------------------------------------------------------------------- #
ALL_COMPONENTS = {*PREDEFINED_RECIPES,
                  *INGREDIENT_TYPES}
