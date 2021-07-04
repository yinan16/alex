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

    def forward(self, x, training):
        x = self.model(x, self.trainable_params, training)
        return x

    @staticmethod
    def get_trainable_params(ckpt=None):
        trainable_params = dict()
        conv_5fo_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        conv_5fo_filters = torch.nn.parameter.Parameter(data=conv_5fo_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['conv_5fo/filters'] = conv_5fo_filters
        batch_normalize_11lk_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        batch_normalize_11lk_mean = torch.nn.parameter.Parameter(data=batch_normalize_11lk_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['batch_normalize_11lk/mean'] = batch_normalize_11lk_mean
        batch_normalize_11lk_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        batch_normalize_11lk_offset = torch.nn.parameter.Parameter(data=batch_normalize_11lk_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['batch_normalize_11lk/offset'] = batch_normalize_11lk_offset
        batch_normalize_11lk_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        batch_normalize_11lk_scale = torch.nn.parameter.Parameter(data=batch_normalize_11lk_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['batch_normalize_11lk/scale'] = batch_normalize_11lk_scale
        batch_normalize_11lk_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        batch_normalize_11lk_variance = torch.nn.parameter.Parameter(data=batch_normalize_11lk_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['batch_normalize_11lk/variance'] = batch_normalize_11lk_variance
        conv_13na_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        conv_13na_filters = torch.nn.parameter.Parameter(data=conv_13na_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['conv_13na/filters'] = conv_13na_filters
        conv_15pq_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        conv_15pq_filters = torch.nn.parameter.Parameter(data=conv_15pq_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['conv_15pq/filters'] = conv_15pq_filters
        dense_19tw_bias_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[1, ]))
        dense_19tw_bias = torch.nn.parameter.Parameter(data=dense_19tw_bias_initializer_zeros_initializer, requires_grad=True)
        trainable_params['dense_19tw/bias'] = dense_19tw_bias
        dense_19tw_weights_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[10, 4096]))
        dense_19tw_weights = torch.nn.parameter.Parameter(data=dense_19tw_weights_initializer_xavier_uniform, requires_grad=True)
        trainable_params['dense_19tw/weights'] = dense_19tw_weights
        return trainable_params
    
    @staticmethod
    def model(input_data, trainable_params, training):
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=trainable_params['conv_5fo/filters'], bias=None, stride=2, padding=[1, 1], dilation=1, groups=1)
        reluu = torch.nn.functional.relu(input=conv_5fo, inplace=False)
        dropout_9ju = torch.nn.functional.dropout(input=reluu, p=0.2, training=training, inplace=False)
        batch_normalize_11lk = torch.nn.functional.batch_norm(input=dropout_9ju, running_mean=trainable_params['batch_normalize_11lk/mean'], running_var=trainable_params['batch_normalize_11lk/variance'], weight=trainable_params['batch_normalize_11lk/scale'], bias=trainable_params['batch_normalize_11lk/offset'], training=training, momentum=0.1, eps=0.001)
        conv_13na = torch.nn.functional.conv2d(input=batch_normalize_11lk, weight=trainable_params['conv_13na/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        conv_15pq = torch.nn.functional.conv2d(input=conv_13na, weight=trainable_params['conv_15pq/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        flatten_17rg = torch.flatten(input=conv_15pq, start_dim=1, end_dim=-1)
        dense_19tw = torch.nn.functional.linear(weight=trainable_params['dense_19tw/weights'], bias=trainable_params['dense_19tw/bias'], input=flatten_17rg)
        d_1 = torch.nn.functional.softmax(input=dense_19tw, dim=None)
        return d_1 
    
    @staticmethod
    def get_loss(trainable_params, inputs):
        cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=inputs[0], input=inputs[1])
        regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['conv_5fo/filters', 'conv_13na/filters', 'conv_15pq/filters', 'dense_19tw/weights'])))
        losses = torch.add(input=[cross_0, regularizer][0], other=[cross_0, regularizer][1])
        return losses 
    
    @staticmethod
    def get_optimizer(trainable_params):
        solver = torch.optim.Adam(params=trainable_params, lr=0.1, betas=(0.9, 0.999), eps=1e-08)
        return solver 
    
    @staticmethod
    def get_scheduler(optimizer):
        solver_decay_exponential_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1, verbose=False)
        return solver_decay_exponential_decay 
    