import torch 
import numpy as np


torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_types = {'float32': torch.float32, 'int8': torch.int8}


class Model(torch.nn.Module):

    def __init__(self, ckpt=None):
        super(Model, self).__init__()
        self.trainable_params = self.get_trainable_params(ckpt)
        self.params = []
        for var in self.trainable_params:
            self.register_parameter(var, self.trainable_params[var])
            self.params.append({'params': self.trainable_params[var]})

    def forward(self, x, trainable_params):
        x = self.model(x, trainable_params)
        return x

    @staticmethod
    def get_trainable_params():
        trainable_params = dict()
        model_block_test_recipe_34im_conv_6gw_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_test_recipe_34im_conv_6gw_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_6gw_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/conv_6gw/filters'] = model_block_test_recipe_34im_conv_6gw_filters
        model_block_test_recipe_34im_batch_normalize_8im_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_batch_normalize_8im_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_batch_normalize_8im_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/batch_normalize_8im/mean'] = model_block_test_recipe_34im_batch_normalize_8im_mean
        model_block_test_recipe_34im_batch_normalize_8im_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_batch_normalize_8im_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_batch_normalize_8im_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/batch_normalize_8im/offset'] = model_block_test_recipe_34im_batch_normalize_8im_offset
        model_block_test_recipe_34im_batch_normalize_8im_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_batch_normalize_8im_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_batch_normalize_8im_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/batch_normalize_8im/scale'] = model_block_test_recipe_34im_batch_normalize_8im_scale
        model_block_test_recipe_34im_batch_normalize_8im_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_batch_normalize_8im_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_batch_normalize_8im_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/batch_normalize_8im/variance'] = model_block_test_recipe_34im_batch_normalize_8im_variance
        model_block_test_recipe_34im_conv_12ms_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_34im_conv_12ms_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_12ms_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/conv_12ms/filters'] = model_block_test_recipe_34im_conv_12ms_filters
        model_block_test_recipe_34im_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_conv_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/conv/mean'] = model_block_test_recipe_34im_conv_mean
        model_block_test_recipe_34im_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_conv_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/conv/offset'] = model_block_test_recipe_34im_conv_offset
        model_block_test_recipe_34im_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_conv_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/conv/scale'] = model_block_test_recipe_34im_conv_scale
        model_block_test_recipe_34im_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_conv_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/conv/variance'] = model_block_test_recipe_34im_conv_variance
        model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters'] = model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/mean'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/offset'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/scale'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance
        model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters'] = model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters
        model_block_test_recipe_34im_resnet_16_33he_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_conv_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/mean'] = model_block_test_recipe_34im_resnet_16_33he_conv_mean
        model_block_test_recipe_34im_resnet_16_33he_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_conv_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/offset'] = model_block_test_recipe_34im_resnet_16_33he_conv_offset
        model_block_test_recipe_34im_resnet_16_33he_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_conv_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/scale'] = model_block_test_recipe_34im_resnet_16_33he_conv_scale
        model_block_test_recipe_34im_resnet_16_33he_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_34im_resnet_16_33he_conv_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_34im_resnet_16_33he_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/variance'] = model_block_test_recipe_34im_resnet_16_33he_conv_variance
        model_block_resnet_16_49xc_0_conv_36kc_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_resnet_16_49xc_0_conv_36kc_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_36kc_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/conv_36kc/filters'] = model_block_resnet_16_49xc_0_conv_36kc_filters
        model_block_resnet_16_49xc_0_batch_normalize_38ms_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_batch_normalize_38ms_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_batch_normalize_38ms_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/mean'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_mean
        model_block_resnet_16_49xc_0_batch_normalize_38ms_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_batch_normalize_38ms_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_batch_normalize_38ms_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/offset'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_offset
        model_block_resnet_16_49xc_0_batch_normalize_38ms_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_batch_normalize_38ms_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_batch_normalize_38ms_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/scale'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_scale
        model_block_resnet_16_49xc_0_batch_normalize_38ms_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_batch_normalize_38ms_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_batch_normalize_38ms_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/variance'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_variance
        model_block_resnet_16_49xc_0_conv_42qy_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_49xc_0_conv_42qy_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_42qy_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/conv_42qy/filters'] = model_block_resnet_16_49xc_0_conv_42qy_filters
        model_block_resnet_16_49xc_0_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_conv_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc_0/conv/mean'] = model_block_resnet_16_49xc_0_conv_mean
        model_block_resnet_16_49xc_0_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_conv_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/conv/offset'] = model_block_resnet_16_49xc_0_conv_offset
        model_block_resnet_16_49xc_0_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_conv_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc_0/conv/scale'] = model_block_resnet_16_49xc_0_conv_scale
        model_block_resnet_16_49xc_0_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_0_conv_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_0_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc_0/conv/variance'] = model_block_resnet_16_49xc_0_conv_variance
        model_block_resnet_16_49xc_conv_36kc_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_resnet_16_49xc_conv_36kc_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_36kc_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/conv_36kc/filters'] = model_block_resnet_16_49xc_conv_36kc_filters
        model_block_resnet_16_49xc_batch_normalize_38ms_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_batch_normalize_38ms_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_batch_normalize_38ms_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/mean'] = model_block_resnet_16_49xc_batch_normalize_38ms_mean
        model_block_resnet_16_49xc_batch_normalize_38ms_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_batch_normalize_38ms_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_batch_normalize_38ms_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/offset'] = model_block_resnet_16_49xc_batch_normalize_38ms_offset
        model_block_resnet_16_49xc_batch_normalize_38ms_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_batch_normalize_38ms_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_batch_normalize_38ms_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/scale'] = model_block_resnet_16_49xc_batch_normalize_38ms_scale
        model_block_resnet_16_49xc_batch_normalize_38ms_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_batch_normalize_38ms_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_batch_normalize_38ms_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/variance'] = model_block_resnet_16_49xc_batch_normalize_38ms_variance
        model_block_resnet_16_49xc_conv_42qy_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_49xc_conv_42qy_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_42qy_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/conv_42qy/filters'] = model_block_resnet_16_49xc_conv_42qy_filters
        model_block_resnet_16_49xc_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_conv_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc/conv/mean'] = model_block_resnet_16_49xc_conv_mean
        model_block_resnet_16_49xc_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_conv_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/conv/offset'] = model_block_resnet_16_49xc_conv_offset
        model_block_resnet_16_49xc_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_conv_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_49xc/conv/scale'] = model_block_resnet_16_49xc_conv_scale
        model_block_resnet_16_49xc_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_49xc_conv_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_49xc_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_49xc/conv/variance'] = model_block_resnet_16_49xc_conv_variance
        model_block_batch_normalize_55dy_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[3, ]))
        model_block_batch_normalize_55dy_mean = torch.nn.parameter.Parameter(data=model_block_batch_normalize_55dy_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_55dy/mean'] = model_block_batch_normalize_55dy_mean
        model_block_batch_normalize_55dy_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[3, ]))
        model_block_batch_normalize_55dy_offset = torch.nn.parameter.Parameter(data=model_block_batch_normalize_55dy_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_55dy/offset'] = model_block_batch_normalize_55dy_offset
        model_block_batch_normalize_55dy_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[3, ]))
        model_block_batch_normalize_55dy_scale = torch.nn.parameter.Parameter(data=model_block_batch_normalize_55dy_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_55dy/scale'] = model_block_batch_normalize_55dy_scale
        model_block_batch_normalize_55dy_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[3, ]))
        model_block_batch_normalize_55dy_variance = torch.nn.parameter.Parameter(data=model_block_batch_normalize_55dy_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_55dy/variance'] = model_block_batch_normalize_55dy_variance
        model_block_conv_57fo_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_conv_57fo_filters = torch.nn.parameter.Parameter(data=model_block_conv_57fo_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_57fo/filters'] = model_block_conv_57fo_filters
        model_block_dense_63lk_bias_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[1, ]))
        model_block_dense_63lk_bias = torch.nn.parameter.Parameter(data=model_block_dense_63lk_bias_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/dense_63lk/bias'] = model_block_dense_63lk_bias
        model_block_dense_63lk_weights_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[10, 14400]))
        model_block_dense_63lk_weights = torch.nn.parameter.Parameter(data=model_block_dense_63lk_weights_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/dense_63lk/weights'] = model_block_dense_63lk_weights
        return trainable_params
    
    @staticmethod
    def model(trainable_params, training, data_block_input_data):
        model_block_test_recipe_34im_conv_6gw = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/test_recipe_34im/conv_6gw/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_34im_batch_normalize_8im = torch.nn.functional.batch_norm(input=model_block_test_recipe_34im_conv_6gw, running_mean=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/mean'], running_var=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/variance'], weight=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/scale'], bias=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_34im_relu_10kc = torch.nn.functional.relu(input=model_block_test_recipe_34im_batch_normalize_8im, inplace=False)
        model_block_test_recipe_34im_conv_12ms = torch.nn.functional.conv2d(input=model_block_test_recipe_34im_relu_10kc, weight=trainable_params['model_block/test_recipe_34im/conv_12ms/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_34im_conv = torch.nn.functional.batch_norm(input=model_block_test_recipe_34im_conv_12ms, running_mean=trainable_params['model_block/test_recipe_34im/conv/mean'], running_var=trainable_params['model_block/test_recipe_34im/conv/variance'], weight=trainable_params['model_block/test_recipe_34im/conv/scale'], bias=trainable_params['model_block/test_recipe_34im/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_34im_add_16qy = torch.add(input=[data_block_input_data, model_block_test_recipe_34im_conv][0], other=[data_block_input_data, model_block_test_recipe_34im_conv][1])
        model_block_test_recipe_34im_relu_18so = torch.nn.functional.relu(input=model_block_test_recipe_34im_add_16qy, inplace=False)
        model_block_test_recipe_34im_resnet_16_33he_conv_20ue = torch.nn.functional.conv2d(input=model_block_test_recipe_34im_relu_18so, weight=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu = torch.nn.functional.batch_norm(input=model_block_test_recipe_34im_resnet_16_33he_conv_20ue, running_mean=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/mean'], running_var=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance'], weight=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/scale'], bias=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_34im_resnet_16_33he_relu_24yk = torch.nn.functional.relu(input=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu, inplace=False)
        model_block_test_recipe_34im_resnet_16_33he_conv_26aa = torch.nn.functional.conv2d(input=model_block_test_recipe_34im_resnet_16_33he_relu_24yk, weight=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_34im_resnet_16_33he_conv = torch.nn.functional.batch_norm(input=model_block_test_recipe_34im_resnet_16_33he_conv_26aa, running_mean=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/mean'], running_var=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/variance'], weight=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/scale'], bias=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_34im_resnet_16_33he_add_30eg = torch.add(input=[model_block_test_recipe_34im_relu_18so, model_block_test_recipe_34im_resnet_16_33he_conv][0], other=[model_block_test_recipe_34im_relu_18so, model_block_test_recipe_34im_resnet_16_33he_conv][1])
        model_block_test_recipe_34im_resnet_16_33he_relu_32gw = torch.nn.functional.relu(input=model_block_test_recipe_34im_resnet_16_33he_add_30eg, inplace=False)
        model_block_resnet_16_49xc_0_conv_36kc = torch.nn.functional.conv2d(input=model_block_test_recipe_34im_resnet_16_33he_relu_32gw, weight=trainable_params['model_block/resnet_16_49xc_0/conv_36kc/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_49xc_0_batch_normalize_38ms = torch.nn.functional.batch_norm(input=model_block_resnet_16_49xc_0_conv_36kc, running_mean=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/mean'], running_var=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/variance'], weight=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/scale'], bias=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_49xc_0_relu_40oi = torch.nn.functional.relu(input=model_block_resnet_16_49xc_0_batch_normalize_38ms, inplace=False)
        model_block_resnet_16_49xc_0_conv_42qy = torch.nn.functional.conv2d(input=model_block_resnet_16_49xc_0_relu_40oi, weight=trainable_params['model_block/resnet_16_49xc_0/conv_42qy/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_49xc_0_conv = torch.nn.functional.batch_norm(input=model_block_resnet_16_49xc_0_conv_42qy, running_mean=trainable_params['model_block/resnet_16_49xc_0/conv/mean'], running_var=trainable_params['model_block/resnet_16_49xc_0/conv/variance'], weight=trainable_params['model_block/resnet_16_49xc_0/conv/scale'], bias=trainable_params['model_block/resnet_16_49xc_0/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_49xc_0_add_46ue = torch.add(input=[model_block_test_recipe_34im_relu_18so, model_block_resnet_16_49xc_0_conv][0], other=[model_block_test_recipe_34im_relu_18so, model_block_resnet_16_49xc_0_conv][1])
        model_block_resnet_16_49xc_0_relu_48wu = torch.nn.functional.relu(input=model_block_resnet_16_49xc_0_add_46ue, inplace=False)
        model_block_resnet_16_49xc_conv_36kc = torch.nn.functional.conv2d(input=model_block_resnet_16_49xc_0_relu_48wu, weight=trainable_params['model_block/resnet_16_49xc/conv_36kc/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_49xc_batch_normalize_38ms = torch.nn.functional.batch_norm(input=model_block_resnet_16_49xc_conv_36kc, running_mean=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/mean'], running_var=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/variance'], weight=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/scale'], bias=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_49xc_relu_40oi = torch.nn.functional.relu(input=model_block_resnet_16_49xc_batch_normalize_38ms, inplace=False)
        model_block_resnet_16_49xc_conv_42qy = torch.nn.functional.conv2d(input=model_block_resnet_16_49xc_relu_40oi, weight=trainable_params['model_block/resnet_16_49xc/conv_42qy/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_49xc_conv = torch.nn.functional.batch_norm(input=model_block_resnet_16_49xc_conv_42qy, running_mean=trainable_params['model_block/resnet_16_49xc/conv/mean'], running_var=trainable_params['model_block/resnet_16_49xc/conv/variance'], weight=trainable_params['model_block/resnet_16_49xc/conv/scale'], bias=trainable_params['model_block/resnet_16_49xc/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_49xc_add_46ue = torch.add(input=[model_block_resnet_16_49xc_0_relu_48wu, model_block_resnet_16_49xc_conv][0], other=[model_block_resnet_16_49xc_0_relu_48wu, model_block_resnet_16_49xc_conv][1])
        model_block_resnet_16_49xc_relu_48wu = torch.nn.functional.relu(input=model_block_resnet_16_49xc_add_46ue, inplace=False)
        model_block_relu_51zs = torch.nn.functional.relu(input=model_block_resnet_16_49xc_relu_48wu, inplace=False)
        model_block_dropout_53bi = torch.nn.functional.dropout(input=model_block_relu_51zs, p=0.2, training=training, inplace=False)
        model_block_batch_normalize_55dy = torch.nn.functional.batch_norm(input=model_block_dropout_53bi, running_mean=trainable_params['model_block/batch_normalize_55dy/mean'], running_var=trainable_params['model_block/batch_normalize_55dy/variance'], weight=trainable_params['model_block/batch_normalize_55dy/scale'], bias=trainable_params['model_block/batch_normalize_55dy/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_conv_57fo = torch.nn.functional.conv2d(input=model_block_batch_normalize_55dy, weight=trainable_params['model_block/conv_57fo/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_max_pool2d_59he = torch.nn.functional.max_pool2d(input=model_block_conv_57fo, kernel_size=3, stride=1, padding=[0, 0])
        model_block_flatten_61ju = torch.flatten(input=model_block_max_pool2d_59he, start_dim=1, end_dim=-1)
        model_block_dense_63lk = torch.nn.functional.linear(weight=trainable_params['model_block/dense_63lk/weights'], bias=trainable_params['model_block/dense_63lk/bias'], input=model_block_flatten_61ju)
        model_block_d_1 = torch.nn.functional.softmax(input=model_block_dense_63lk, dim=None)
        return model_block_d_1 
    
    @staticmethod
    def get_loss(inputs, trainable_params):
        loss_block_cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=inputs[0], input=inputs[1])
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/test_recipe_34im/conv_6gw/filters', 'model_block/test_recipe_34im/conv_12ms/filters', 'model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters', 'model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters', 'model_block/resnet_16_49xc_0/conv_36kc/filters', 'model_block/resnet_16_49xc_0/conv_42qy/filters', 'model_block/resnet_16_49xc/conv_36kc/filters', 'model_block/resnet_16_49xc/conv_42qy/filters', 'model_block/conv_57fo/filters', 'model_block/dense_63lk/weights'])))
        loss_block_losses = torch.add(input=[loss_block_cross_0, loss_block_regularizer][0], other=[loss_block_cross_0, loss_block_regularizer][1])
        return loss_block_losses 
    
    @staticmethod
    def get_optimizer(trainable_params):
        optimizer_block_solver = torch.optim.Adam(params=trainable_params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        return optimizer_block_solver 
    
    @staticmethod
    def get_scheduler(optimizer):
        optimizer_block_solver_decay_exponential_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1, verbose=False)
        return optimizer_block_solver_decay_exponential_decay 
    
from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small3.yml", None, None)

ckpt = C.load()

model = Model(ckpt)

model.to(device)

trainable_params = model.trainable_params
optimizer = model.get_optimizer(model.params)

learning_rate = model.get_scheduler(optimizer)


def inference(trainable_params, data_block_input_data):
    
    model.training=False
    training = model.training
    
    preds = torch.max(model(trainable_params, training, data_block_input_data), 1)
    preds = preds[1]
    return preds
    
def evaluation(labels, data_block_input_data, trainable_params):
    
    preds = inference(trainable_params, data_block_input_data)
    
    model.training=False
    training = model.training
    
    total = labels.size(0)
    matches = (preds == labels).sum().item()
    perf = matches / total
    
    loss = model.get_loss([labels, preds], trainable_params)
    return perf, loss
    
    
def train(labels, data_block_input_data, trainable_params):
    
    optimizer.zero_grad()
    model.training=True
    training = model.training
    preds = model(trainable_params, training, data_block_input_data)
    loss = model.get_loss([labels, preds], trainable_params)
    loss.backward()
    
    
def loop(trainloader, test_inputs, test_labels):
    
    for epoch in range(90):
    
        for i, data in enumerate(trainloader, 0):
    
            inputs, labels = data
    
            inputs = inputs.to(device)
            labels = labels.to(device)
            train(labels, inputs, trainable_params)
            optimizer.step()
    
            if i % 500 == 499:
                results = evaluation(val_labels, val_inputs, trainable_params)
                print(results)
                
        learning_rate.step()
    print('Finished Training')
    
    

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor()])
# ,
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=10000,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
inputs = images.to(device)
print(device)
# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

val_inputs, val_labels = iter(valloader).next()

loop(trainloader, val_inputs, val_labels)


