import torch 
import numpy as np


torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_types = {'float32': torch.float32, 'int8': torch.int8}


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.trainable_params = self.get_trainable_params(input_shape, True)
        self.params = []
        for var in self.trainable_params:
            self.register_parameter(var, self.trainable_params[var])
            self.params.append({'params': self.trainable_params[var]})
    def forward(self, x, training):
        x = self.model(x, self.trainable_params, training)
        return x

    @staticmethod
    def get_trainable_params(input_shape, training):
        trainable_params = dict()
        input_data = torch.zeros(input_shape)
        conv_5fo_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(input_data.size())[1], 3, 3, dtype=torch_types['float32']))
        conv_5fo_filters = torch.nn.parameter.Parameter(data=conv_5fo_filters_init, requires_grad=True)
        trainable_params['conv_5fo/filters'] = conv_5fo_filters
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=conv_5fo_filters, bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        relu_7he = torch.nn.functional.relu(input=conv_5fo, inplace=False)
        dropout_9ju = torch.nn.functional.dropout(input=relu_7he, p=0.2, training=training, inplace=False)
        batch_normalize_11lk_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(dropout_9ju.size())[1], dtype=torch_types['float32']))
        batch_normalize_11lk_mean = torch.nn.parameter.Parameter(data=batch_normalize_11lk_mean_init, requires_grad=False)
        trainable_params['batch_normalize_11lk/mean'] = batch_normalize_11lk_mean
        batch_normalize_11lk_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(dropout_9ju.size())[1], dtype=torch_types['float32']))
        batch_normalize_11lk_variance = torch.nn.parameter.Parameter(data=batch_normalize_11lk_variance_init, requires_grad=False)
        trainable_params['batch_normalize_11lk/variance'] = batch_normalize_11lk_variance
        batch_normalize_11lk_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(dropout_9ju.size())[1], dtype=torch_types['float32']))
        batch_normalize_11lk_offset = torch.nn.parameter.Parameter(data=batch_normalize_11lk_offset_init, requires_grad=True)
        trainable_params['batch_normalize_11lk/offset'] = batch_normalize_11lk_offset
        batch_normalize_11lk_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(dropout_9ju.size())[1], dtype=torch_types['float32']))
        batch_normalize_11lk_scale = torch.nn.parameter.Parameter(data=batch_normalize_11lk_scale_init, requires_grad=True)
        trainable_params['batch_normalize_11lk/scale'] = batch_normalize_11lk_scale
        batch_normalize_11lk = torch.nn.functional.batch_norm(input=dropout_9ju, running_mean=batch_normalize_11lk_mean, running_var=batch_normalize_11lk_variance, weight=batch_normalize_11lk_scale, bias=batch_normalize_11lk_offset, training=training, momentum=0.1, eps=0.001)
        conv_13na_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(batch_normalize_11lk.size())[1], 3, 3, dtype=torch_types['float32']))
        conv_13na_filters = torch.nn.parameter.Parameter(data=conv_13na_filters_init, requires_grad=True)
        trainable_params['conv_13na/filters'] = conv_13na_filters
        conv_13na = torch.nn.functional.conv2d(input=batch_normalize_11lk, weight=conv_13na_filters, bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        batch_normalize_15pq_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(conv_13na.size())[1], dtype=torch_types['float32']))
        batch_normalize_15pq_mean = torch.nn.parameter.Parameter(data=batch_normalize_15pq_mean_init, requires_grad=False)
        trainable_params['batch_normalize_15pq/mean'] = batch_normalize_15pq_mean
        batch_normalize_15pq_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(conv_13na.size())[1], dtype=torch_types['float32']))
        batch_normalize_15pq_variance = torch.nn.parameter.Parameter(data=batch_normalize_15pq_variance_init, requires_grad=False)
        trainable_params['batch_normalize_15pq/variance'] = batch_normalize_15pq_variance
        batch_normalize_15pq_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(conv_13na.size())[1], dtype=torch_types['float32']))
        batch_normalize_15pq_offset = torch.nn.parameter.Parameter(data=batch_normalize_15pq_offset_init, requires_grad=True)
        trainable_params['batch_normalize_15pq/offset'] = batch_normalize_15pq_offset
        batch_normalize_15pq_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(conv_13na.size())[1], dtype=torch_types['float32']))
        batch_normalize_15pq_scale = torch.nn.parameter.Parameter(data=batch_normalize_15pq_scale_init, requires_grad=True)
        trainable_params['batch_normalize_15pq/scale'] = batch_normalize_15pq_scale
        batch_normalize_15pq = torch.nn.functional.batch_norm(input=conv_13na, running_mean=batch_normalize_15pq_mean, running_var=batch_normalize_15pq_variance, weight=batch_normalize_15pq_scale, bias=batch_normalize_15pq_offset, training=training, momentum=0.1, eps=0.001)
        conv_17rg_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(batch_normalize_15pq.size())[1], 3, 3, dtype=torch_types['float32']))
        conv_17rg_filters = torch.nn.parameter.Parameter(data=conv_17rg_filters_init, requires_grad=True)
        trainable_params['conv_17rg/filters'] = conv_17rg_filters
        conv_17rg = torch.nn.functional.conv2d(input=batch_normalize_15pq, weight=conv_17rg_filters, bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_conv_19tw_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(conv_17rg.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_16_32gw_conv_19tw_filters = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_19tw_filters_init, requires_grad=True)
        trainable_params['resnet_16_32gw/conv_19tw/filters'] = resnet_16_32gw_conv_19tw_filters
        resnet_16_32gw_conv_19tw = torch.nn.functional.conv2d(input=conv_17rg, weight=resnet_16_32gw_conv_19tw_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_batch_normalize_21vm_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_32gw_conv_19tw.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_batch_normalize_21vm_mean = torch.nn.parameter.Parameter(data=resnet_16_32gw_batch_normalize_21vm_mean_init, requires_grad=False)
        trainable_params['resnet_16_32gw/batch_normalize_21vm/mean'] = resnet_16_32gw_batch_normalize_21vm_mean
        resnet_16_32gw_batch_normalize_21vm_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_32gw_conv_19tw.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_batch_normalize_21vm_variance = torch.nn.parameter.Parameter(data=resnet_16_32gw_batch_normalize_21vm_variance_init, requires_grad=False)
        trainable_params['resnet_16_32gw/batch_normalize_21vm/variance'] = resnet_16_32gw_batch_normalize_21vm_variance
        resnet_16_32gw_batch_normalize_21vm_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_32gw_conv_19tw.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_batch_normalize_21vm_offset = torch.nn.parameter.Parameter(data=resnet_16_32gw_batch_normalize_21vm_offset_init, requires_grad=True)
        trainable_params['resnet_16_32gw/batch_normalize_21vm/offset'] = resnet_16_32gw_batch_normalize_21vm_offset
        resnet_16_32gw_batch_normalize_21vm_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_32gw_conv_19tw.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_batch_normalize_21vm_scale = torch.nn.parameter.Parameter(data=resnet_16_32gw_batch_normalize_21vm_scale_init, requires_grad=True)
        trainable_params['resnet_16_32gw/batch_normalize_21vm/scale'] = resnet_16_32gw_batch_normalize_21vm_scale
        resnet_16_32gw_batch_normalize_21vm = torch.nn.functional.batch_norm(input=resnet_16_32gw_conv_19tw, running_mean=resnet_16_32gw_batch_normalize_21vm_mean, running_var=resnet_16_32gw_batch_normalize_21vm_variance, weight=resnet_16_32gw_batch_normalize_21vm_scale, bias=resnet_16_32gw_batch_normalize_21vm_offset, training=training, momentum=0.1, eps=0.001)
        resnet_16_32gw_relu_23xc = torch.nn.functional.relu(input=resnet_16_32gw_batch_normalize_21vm, inplace=False)
        resnet_16_32gw_conv_25zs_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(resnet_16_32gw_relu_23xc.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_16_32gw_conv_25zs_filters = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_25zs_filters_init, requires_grad=True)
        trainable_params['resnet_16_32gw/conv_25zs/filters'] = resnet_16_32gw_conv_25zs_filters
        resnet_16_32gw_conv_25zs = torch.nn.functional.conv2d(input=resnet_16_32gw_relu_23xc, weight=resnet_16_32gw_conv_25zs_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_conv_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_32gw_conv_25zs.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_conv_mean = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_mean_init, requires_grad=False)
        trainable_params['resnet_16_32gw/conv/mean'] = resnet_16_32gw_conv_mean
        resnet_16_32gw_conv_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_32gw_conv_25zs.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_conv_variance = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_variance_init, requires_grad=False)
        trainable_params['resnet_16_32gw/conv/variance'] = resnet_16_32gw_conv_variance
        resnet_16_32gw_conv_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_32gw_conv_25zs.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_conv_offset = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_offset_init, requires_grad=True)
        trainable_params['resnet_16_32gw/conv/offset'] = resnet_16_32gw_conv_offset
        resnet_16_32gw_conv_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_32gw_conv_25zs.size())[1], dtype=torch_types['float32']))
        resnet_16_32gw_conv_scale = torch.nn.parameter.Parameter(data=resnet_16_32gw_conv_scale_init, requires_grad=True)
        trainable_params['resnet_16_32gw/conv/scale'] = resnet_16_32gw_conv_scale
        resnet_16_32gw_conv = torch.nn.functional.batch_norm(input=resnet_16_32gw_conv_25zs, running_mean=resnet_16_32gw_conv_mean, running_var=resnet_16_32gw_conv_variance, weight=resnet_16_32gw_conv_scale, bias=resnet_16_32gw_conv_offset, training=training, momentum=0.1, eps=0.001)
        resnet_16_32gw_add_29dy = torch.add(input=[conv_17rg, resnet_16_32gw_conv][0], other=[conv_17rg, resnet_16_32gw_conv][1])
        resnet_16_32gw_relu_31fo = torch.nn.functional.relu(input=resnet_16_32gw_add_29dy, inplace=False)
        flatten_34im = torch.flatten(input=resnet_16_32gw_relu_31fo, start_dim=1, end_dim=-1)
        dense_36kc_weights_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(10, list(flatten_34im.size())[1], dtype=torch_types['float32']))
        dense_36kc_weights = torch.nn.parameter.Parameter(data=dense_36kc_weights_init, requires_grad=True)
        trainable_params['dense_36kc/weights'] = dense_36kc_weights
        dense_36kc_bias_init = torch.nn.init.zeros_(tensor=torch.empty(1, dtype=torch_types['float32']))
        dense_36kc_bias = torch.nn.parameter.Parameter(data=dense_36kc_bias_init, requires_grad=True)
        trainable_params['dense_36kc/bias'] = dense_36kc_bias
        return trainable_params
    
    @staticmethod
    def model(input_data, trainable_params, training):
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=trainable_params['conv_5fo/filters'], bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        relu_7he = torch.nn.functional.relu(input=conv_5fo, inplace=False)
        dropout_9ju = torch.nn.functional.dropout(input=relu_7he, p=0.2, training=training, inplace=False)
        batch_normalize_11lk = torch.nn.functional.batch_norm(input=dropout_9ju, running_mean=trainable_params['batch_normalize_11lk/mean'], running_var=trainable_params['batch_normalize_11lk/variance'], weight=trainable_params['batch_normalize_11lk/scale'], bias=trainable_params['batch_normalize_11lk/offset'], training=training, momentum=0.1, eps=0.001)
        conv_13na = torch.nn.functional.conv2d(input=batch_normalize_11lk, weight=trainable_params['conv_13na/filters'], bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        batch_normalize_15pq = torch.nn.functional.batch_norm(input=conv_13na, running_mean=trainable_params['batch_normalize_15pq/mean'], running_var=trainable_params['batch_normalize_15pq/variance'], weight=trainable_params['batch_normalize_15pq/scale'], bias=trainable_params['batch_normalize_15pq/offset'], training=training, momentum=0.1, eps=0.001)
        conv_17rg = torch.nn.functional.conv2d(input=batch_normalize_15pq, weight=trainable_params['conv_17rg/filters'], bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_conv_19tw = torch.nn.functional.conv2d(input=conv_17rg, weight=trainable_params['resnet_16_32gw/conv_19tw/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_batch_normalize_21vm = torch.nn.functional.batch_norm(input=resnet_16_32gw_conv_19tw, running_mean=trainable_params['resnet_16_32gw/batch_normalize_21vm/mean'], running_var=trainable_params['resnet_16_32gw/batch_normalize_21vm/variance'], weight=trainable_params['resnet_16_32gw/batch_normalize_21vm/scale'], bias=trainable_params['resnet_16_32gw/batch_normalize_21vm/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_32gw_relu_23xc = torch.nn.functional.relu(input=resnet_16_32gw_batch_normalize_21vm, inplace=False)
        resnet_16_32gw_conv_25zs = torch.nn.functional.conv2d(input=resnet_16_32gw_relu_23xc, weight=trainable_params['resnet_16_32gw/conv_25zs/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_32gw_conv = torch.nn.functional.batch_norm(input=resnet_16_32gw_conv_25zs, running_mean=trainable_params['resnet_16_32gw/conv/mean'], running_var=trainable_params['resnet_16_32gw/conv/variance'], weight=trainable_params['resnet_16_32gw/conv/scale'], bias=trainable_params['resnet_16_32gw/conv/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_32gw_add_29dy = torch.add(input=[conv_17rg, resnet_16_32gw_conv][0], other=[conv_17rg, resnet_16_32gw_conv][1])
        resnet_16_32gw_relu_31fo = torch.nn.functional.relu(input=resnet_16_32gw_add_29dy, inplace=False)
        flatten_34im = torch.flatten(input=resnet_16_32gw_relu_31fo, start_dim=1, end_dim=-1)
        dense_36kc = torch.nn.functional.linear(input=flatten_34im, weight=trainable_params['dense_36kc/weights'], bias=trainable_params['dense_36kc/bias'])
        d_1 = torch.nn.functional.softmax(input=dense_36kc)
        return d_1 
    
    @staticmethod
    def get_loss(d_1, labels):
        cross_entropy_40oi = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=[labels, d_1][0], input=[labels, d_1][1])
        losses = torch.nn.functional.sigmoid(input=cross_entropy_40oi)
        return losses 
    
    @staticmethod
    def get_optimizer(trainable_params):
        solver = {'optimizer': torch.optim.Adam(params=trainable_params, lr=0.1, betas=(0.9, 0.999), eps=1e-08), 'learning_rate': lambda optimizer: 0.1}
        return solver 
    