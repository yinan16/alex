import torch 
import numpy as np


torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_types = {'float32': torch.float32, 'int8': torch.int8}


class Model(torch.nn.Module):

    def __init__(self, ckpt=None):
        super(Model, self).__init__()
        self.trainable_params = self.get_trainable_params()
        self.params = []
        for var in self.trainable_params:
            self.register_parameter(var, self.trainable_params[var])
            self.params.append({'params': self.trainable_params[var]})

    def forward(self, trainable_params, data_block_input_data, training):
        x = self.model(trainable_params, data_block_input_data, training)
        return x

    @staticmethod
    def get_trainable_params():
        trainable_params = dict()
        model_block_conv_6gw_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_conv_6gw_filters = torch.nn.parameter.Parameter(data=model_block_conv_6gw_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
        model_block_resnet_16_21vm_0_conv_8im_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_21vm_0_conv_8im_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_8im_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/conv_8im/filters'] = model_block_resnet_16_21vm_0_conv_8im_filters
        model_block_resnet_16_21vm_0_batch_normalize_10kc_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_batch_normalize_10kc_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_batch_normalize_10kc_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/mean'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_mean
        model_block_resnet_16_21vm_0_batch_normalize_10kc_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_batch_normalize_10kc_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_batch_normalize_10kc_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/offset'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_offset
        model_block_resnet_16_21vm_0_batch_normalize_10kc_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_batch_normalize_10kc_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_batch_normalize_10kc_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/scale'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_scale
        model_block_resnet_16_21vm_0_batch_normalize_10kc_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_batch_normalize_10kc_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_batch_normalize_10kc_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/variance'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_variance
        model_block_resnet_16_21vm_0_conv_14oi_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_21vm_0_conv_14oi_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_14oi_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/conv_14oi/filters'] = model_block_resnet_16_21vm_0_conv_14oi_filters
        model_block_resnet_16_21vm_0_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_conv_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm_0/conv/mean'] = model_block_resnet_16_21vm_0_conv_mean
        model_block_resnet_16_21vm_0_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_conv_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/conv/offset'] = model_block_resnet_16_21vm_0_conv_offset
        model_block_resnet_16_21vm_0_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_conv_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm_0/conv/scale'] = model_block_resnet_16_21vm_0_conv_scale
        model_block_resnet_16_21vm_0_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_0_conv_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_0_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm_0/conv/variance'] = model_block_resnet_16_21vm_0_conv_variance
        model_block_resnet_16_21vm_conv_8im_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_21vm_conv_8im_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_8im_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/conv_8im/filters'] = model_block_resnet_16_21vm_conv_8im_filters
        model_block_resnet_16_21vm_batch_normalize_10kc_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_batch_normalize_10kc_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_batch_normalize_10kc_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/mean'] = model_block_resnet_16_21vm_batch_normalize_10kc_mean
        model_block_resnet_16_21vm_batch_normalize_10kc_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_batch_normalize_10kc_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_batch_normalize_10kc_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/offset'] = model_block_resnet_16_21vm_batch_normalize_10kc_offset
        model_block_resnet_16_21vm_batch_normalize_10kc_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_batch_normalize_10kc_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_batch_normalize_10kc_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/scale'] = model_block_resnet_16_21vm_batch_normalize_10kc_scale
        model_block_resnet_16_21vm_batch_normalize_10kc_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_batch_normalize_10kc_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_batch_normalize_10kc_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/variance'] = model_block_resnet_16_21vm_batch_normalize_10kc_variance
        model_block_resnet_16_21vm_conv_14oi_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_resnet_16_21vm_conv_14oi_filters = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_14oi_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/conv_14oi/filters'] = model_block_resnet_16_21vm_conv_14oi_filters
        model_block_resnet_16_21vm_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_conv_mean = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm/conv/mean'] = model_block_resnet_16_21vm_conv_mean
        model_block_resnet_16_21vm_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_conv_offset = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/conv/offset'] = model_block_resnet_16_21vm_conv_offset
        model_block_resnet_16_21vm_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_conv_scale = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/resnet_16_21vm/conv/scale'] = model_block_resnet_16_21vm_conv_scale
        model_block_resnet_16_21vm_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_resnet_16_21vm_conv_variance = torch.nn.parameter.Parameter(data=model_block_resnet_16_21vm_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/resnet_16_21vm/conv/variance'] = model_block_resnet_16_21vm_conv_variance
        model_block_test_recipe_51zs_conv_23xc_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_51zs_conv_23xc_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_23xc_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/conv_23xc/filters'] = model_block_test_recipe_51zs_conv_23xc_filters
        model_block_test_recipe_51zs_batch_normalize_25zs_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_batch_normalize_25zs_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_batch_normalize_25zs_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/mean'] = model_block_test_recipe_51zs_batch_normalize_25zs_mean
        model_block_test_recipe_51zs_batch_normalize_25zs_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_batch_normalize_25zs_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_batch_normalize_25zs_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/offset'] = model_block_test_recipe_51zs_batch_normalize_25zs_offset
        model_block_test_recipe_51zs_batch_normalize_25zs_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_batch_normalize_25zs_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_batch_normalize_25zs_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/scale'] = model_block_test_recipe_51zs_batch_normalize_25zs_scale
        model_block_test_recipe_51zs_batch_normalize_25zs_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_batch_normalize_25zs_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_batch_normalize_25zs_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/variance'] = model_block_test_recipe_51zs_batch_normalize_25zs_variance
        model_block_test_recipe_51zs_conv_29dy_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_51zs_conv_29dy_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_29dy_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/conv_29dy/filters'] = model_block_test_recipe_51zs_conv_29dy_filters
        model_block_test_recipe_51zs_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_conv_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/conv/mean'] = model_block_test_recipe_51zs_conv_mean
        model_block_test_recipe_51zs_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_conv_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/conv/offset'] = model_block_test_recipe_51zs_conv_offset
        model_block_test_recipe_51zs_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_conv_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/conv/scale'] = model_block_test_recipe_51zs_conv_scale
        model_block_test_recipe_51zs_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_conv_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/conv/variance'] = model_block_test_recipe_51zs_conv_variance
        model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/mean'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/offset'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/scale'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/variance'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance
        model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters
        model_block_test_recipe_51zs_resnet_16_50yk_conv_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_mean = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/mean'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_mean
        model_block_test_recipe_51zs_resnet_16_50yk_conv_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_offset = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/offset'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_offset
        model_block_test_recipe_51zs_resnet_16_50yk_conv_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_scale = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/scale'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_scale
        model_block_test_recipe_51zs_resnet_16_50yk_conv_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_test_recipe_51zs_resnet_16_50yk_conv_variance = torch.nn.parameter.Parameter(data=model_block_test_recipe_51zs_resnet_16_50yk_conv_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/variance'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_variance
        model_block_conv_53bi_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_conv_53bi_filters = torch.nn.parameter.Parameter(data=model_block_conv_53bi_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_53bi/filters'] = model_block_conv_53bi_filters
        model_block_batch_normalize_59he_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_59he_mean = torch.nn.parameter.Parameter(data=model_block_batch_normalize_59he_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_59he/mean'] = model_block_batch_normalize_59he_mean
        model_block_batch_normalize_59he_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_59he_offset = torch.nn.parameter.Parameter(data=model_block_batch_normalize_59he_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_59he/offset'] = model_block_batch_normalize_59he_offset
        model_block_batch_normalize_59he_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_59he_scale = torch.nn.parameter.Parameter(data=model_block_batch_normalize_59he_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_59he/scale'] = model_block_batch_normalize_59he_scale
        model_block_batch_normalize_59he_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_59he_variance = torch.nn.parameter.Parameter(data=model_block_batch_normalize_59he_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_59he/variance'] = model_block_batch_normalize_59he_variance
        model_block_conv_61ju_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_conv_61ju_filters = torch.nn.parameter.Parameter(data=model_block_conv_61ju_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_61ju/filters'] = model_block_conv_61ju_filters
        model_block_dense_67pq_bias_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[1, ]))
        model_block_dense_67pq_bias = torch.nn.parameter.Parameter(data=model_block_dense_67pq_bias_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/dense_67pq/bias'] = model_block_dense_67pq_bias
        model_block_dense_67pq_weights_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[10, 14400]))
        model_block_dense_67pq_weights = torch.nn.parameter.Parameter(data=model_block_dense_67pq_weights_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/dense_67pq/weights'] = model_block_dense_67pq_weights
        return trainable_params
    
    @staticmethod
    def model(trainable_params, data_block_input_data, training):
        model_block_conv_6gw = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/conv_6gw/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_21vm_0_conv_8im = torch.nn.functional.conv2d(input=model_block_conv_6gw, weight=trainable_params['model_block/resnet_16_21vm_0/conv_8im/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_21vm_0_batch_normalize_10kc = torch.nn.functional.batch_norm(input=model_block_resnet_16_21vm_0_conv_8im, running_mean=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/mean'], running_var=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/variance'], weight=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/scale'], bias=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_21vm_0_relu_12ms = torch.nn.functional.relu(input=model_block_resnet_16_21vm_0_batch_normalize_10kc, inplace=False)
        model_block_resnet_16_21vm_0_conv_14oi = torch.nn.functional.conv2d(input=model_block_resnet_16_21vm_0_relu_12ms, weight=trainable_params['model_block/resnet_16_21vm_0/conv_14oi/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_21vm_0_conv = torch.nn.functional.batch_norm(input=model_block_resnet_16_21vm_0_conv_14oi, running_mean=trainable_params['model_block/resnet_16_21vm_0/conv/mean'], running_var=trainable_params['model_block/resnet_16_21vm_0/conv/variance'], weight=trainable_params['model_block/resnet_16_21vm_0/conv/scale'], bias=trainable_params['model_block/resnet_16_21vm_0/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_21vm_0_add_18so = torch.add(input=[model_block_conv_6gw, model_block_resnet_16_21vm_0_conv][0], other=[model_block_conv_6gw, model_block_resnet_16_21vm_0_conv][1])
        model_block_resnet_16_21vm_0_relu_20ue = torch.nn.functional.relu(input=model_block_resnet_16_21vm_0_add_18so, inplace=False)
        model_block_resnet_16_21vm_conv_8im = torch.nn.functional.conv2d(input=model_block_resnet_16_21vm_0_relu_20ue, weight=trainable_params['model_block/resnet_16_21vm/conv_8im/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_21vm_batch_normalize_10kc = torch.nn.functional.batch_norm(input=model_block_resnet_16_21vm_conv_8im, running_mean=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/mean'], running_var=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/variance'], weight=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/scale'], bias=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_21vm_relu_12ms = torch.nn.functional.relu(input=model_block_resnet_16_21vm_batch_normalize_10kc, inplace=False)
        model_block_resnet_16_21vm_conv_14oi = torch.nn.functional.conv2d(input=model_block_resnet_16_21vm_relu_12ms, weight=trainable_params['model_block/resnet_16_21vm/conv_14oi/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_resnet_16_21vm_conv = torch.nn.functional.batch_norm(input=model_block_resnet_16_21vm_conv_14oi, running_mean=trainable_params['model_block/resnet_16_21vm/conv/mean'], running_var=trainable_params['model_block/resnet_16_21vm/conv/variance'], weight=trainable_params['model_block/resnet_16_21vm/conv/scale'], bias=trainable_params['model_block/resnet_16_21vm/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_resnet_16_21vm_add_18so = torch.add(input=[model_block_resnet_16_21vm_0_relu_20ue, model_block_resnet_16_21vm_conv][0], other=[model_block_resnet_16_21vm_0_relu_20ue, model_block_resnet_16_21vm_conv][1])
        model_block_resnet_16_21vm_relu_20ue = torch.nn.functional.relu(input=model_block_resnet_16_21vm_add_18so, inplace=False)
        model_block_test_recipe_51zs_conv_23xc = torch.nn.functional.conv2d(input=model_block_resnet_16_21vm_relu_20ue, weight=trainable_params['model_block/test_recipe_51zs/conv_23xc/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_51zs_batch_normalize_25zs = torch.nn.functional.batch_norm(input=model_block_test_recipe_51zs_conv_23xc, running_mean=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/mean'], running_var=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/variance'], weight=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/scale'], bias=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_51zs_relu_27bi = torch.nn.functional.relu(input=model_block_test_recipe_51zs_batch_normalize_25zs, inplace=False)
        model_block_test_recipe_51zs_conv_29dy = torch.nn.functional.conv2d(input=model_block_test_recipe_51zs_relu_27bi, weight=trainable_params['model_block/test_recipe_51zs/conv_29dy/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_51zs_conv = torch.nn.functional.batch_norm(input=model_block_test_recipe_51zs_conv_29dy, running_mean=trainable_params['model_block/test_recipe_51zs/conv/mean'], running_var=trainable_params['model_block/test_recipe_51zs/conv/variance'], weight=trainable_params['model_block/test_recipe_51zs/conv/scale'], bias=trainable_params['model_block/test_recipe_51zs/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_51zs_add_33he = torch.add(input=[model_block_resnet_16_21vm_relu_20ue, model_block_test_recipe_51zs_conv][0], other=[model_block_resnet_16_21vm_relu_20ue, model_block_test_recipe_51zs_conv][1])
        model_block_test_recipe_51zs_relu_35ju = torch.nn.functional.relu(input=model_block_test_recipe_51zs_add_33he, inplace=False)
        model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk = torch.nn.functional.conv2d(input=model_block_test_recipe_51zs_relu_35ju, weight=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na = torch.nn.functional.batch_norm(input=model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk, running_mean=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/mean'], running_var=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/variance'], weight=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/scale'], bias=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_51zs_resnet_16_50yk_relu_41pq = torch.nn.functional.relu(input=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na, inplace=False)
        model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg = torch.nn.functional.conv2d(input=model_block_test_recipe_51zs_resnet_16_50yk_relu_41pq, weight=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_test_recipe_51zs_resnet_16_50yk_conv = torch.nn.functional.batch_norm(input=model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg, running_mean=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/mean'], running_var=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/variance'], weight=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/scale'], bias=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_test_recipe_51zs_resnet_16_50yk_add_47vm = torch.add(input=[model_block_test_recipe_51zs_relu_35ju, model_block_test_recipe_51zs_resnet_16_50yk_conv][0], other=[model_block_test_recipe_51zs_relu_35ju, model_block_test_recipe_51zs_resnet_16_50yk_conv][1])
        model_block_test_recipe_51zs_resnet_16_50yk_relu_49xc = torch.nn.functional.relu(input=model_block_test_recipe_51zs_resnet_16_50yk_add_47vm, inplace=False)
        model_block_conv_53bi = torch.nn.functional.conv2d(input=model_block_test_recipe_51zs_resnet_16_50yk_relu_49xc, weight=trainable_params['model_block/conv_53bi/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_relu_55dy = torch.nn.functional.relu(input=model_block_conv_53bi, inplace=False)
        model_block_dropout_57fo = torch.nn.functional.dropout(input=model_block_relu_55dy, p=0.2, training=training, inplace=False)
        model_block_batch_normalize_59he = torch.nn.functional.batch_norm(input=model_block_dropout_57fo, running_mean=trainable_params['model_block/batch_normalize_59he/mean'], running_var=trainable_params['model_block/batch_normalize_59he/variance'], weight=trainable_params['model_block/batch_normalize_59he/scale'], bias=trainable_params['model_block/batch_normalize_59he/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_conv_61ju = torch.nn.functional.conv2d(input=model_block_batch_normalize_59he, weight=trainable_params['model_block/conv_61ju/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_max_pool2d_63lk = torch.nn.functional.max_pool2d(input=model_block_conv_61ju, kernel_size=3, stride=1, padding=[0, 0])
        model_block_flatten_65na = torch.flatten(input=model_block_max_pool2d_63lk, start_dim=1, end_dim=-1)
        model_block_dense_67pq = torch.nn.functional.linear(weight=trainable_params['model_block/dense_67pq/weights'], bias=trainable_params['model_block/dense_67pq/bias'], input=model_block_flatten_65na)
        model_block_output = torch.nn.functional.softmax(input=model_block_dense_67pq, dim=None)
        return model_block_output
    
    @staticmethod
    def get_loss(data_block_labels, trainable_params, model_block_output):
        loss_block_cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=[data_block_labels, model_block_output][0], input=[data_block_labels, model_block_output][1])
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_6gw/filters', 'model_block/resnet_16_21vm_0/conv_8im/filters', 'model_block/resnet_16_21vm_0/conv_14oi/filters', 'model_block/resnet_16_21vm/conv_8im/filters', 'model_block/resnet_16_21vm/conv_14oi/filters', 'model_block/test_recipe_51zs/conv_23xc/filters', 'model_block/test_recipe_51zs/conv_29dy/filters', 'model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters', 'model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters', 'model_block/conv_53bi/filters', 'model_block/conv_61ju/filters', 'model_block/dense_67pq/weights'])))
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

C = Checkpoint("examples/configs/small3.yml", pytorch, None, None)

ckpt = C.load()

model = Model(ckpt)

model.to(device)

trainable_params = model.trainable_params
optimizer = model.get_optimizer(model.params)

learning_rate = model.get_scheduler(optimizer)

probes = dict()

def inference(trainable_params, data_block_input_data):
    
    model.training=False
    training = model.training
    
    preds = torch.max(model(trainable_params, data_block_input_data, training), 1)
    preds = preds[1]
    return preds
    
def evaluation(data_block_labels, trainable_params, labels, data_block_input_data):
    
    preds = inference(trainable_params, data_block_input_data)
    
    model.training=False
    training = model.training
    
    gt = labels
    total = gt.size(0)
    matches = (preds == gt).sum().item()
    perf = matches / total
    
    loss = model.get_loss(data_block_labels, trainable_params, preds)
    return perf, loss
    
    
def train(data_block_labels, trainable_params, data_block_input_data):
    
    optimizer.zero_grad()
    model.training=True
    training = model.training
    preds = model(trainable_params, data_block_input_data, training)
    loss = model.get_loss(data_block_labels, trainable_params, preds)
    loss.backward()
    
    
def loop(trainable_params, val_labels, val_inputs):
    
    for epoch in range(90):
        i = 0
        for data in trainloader:
            inputs, labels = data
    
            inputs = inputs.to(device)
            labels = labels.to(device)
            train(labels, trainable_params, inputs)
            optimizer.step()
    
            if i % 500 == 499:
                results = evaluation(val_labels, trainable_params, labels, val_inputs)
                print("Epoch:", epoch, results)
                
            i += 1
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
valloader = torch.utils.data.DataLoader(valset, batch_size=1000,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()
inputs = images.to(device)
print(device)

val_inputs, val_labels = iter(valloader).next()

val_inputs = val_inputs.to(device)
val_labels = val_labels.to(device)

loop(trainable_params, val_labels, val_inputs)

