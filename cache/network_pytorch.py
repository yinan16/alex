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
        conv_5fo_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(input_data.size())[1], 3, 3, dtype=torch_types['float32']))
        conv_5fo_filters = torch.nn.parameter.Parameter(data=conv_5fo_filters_init, requires_grad=True)
        trainable_params['conv_5fo/filters'] = conv_5fo_filters
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=conv_5fo_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        batch_normalize_7he_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(conv_5fo.size())[1], dtype=torch_types['float32']))
        batch_normalize_7he_mean = torch.nn.parameter.Parameter(data=batch_normalize_7he_mean_init, requires_grad=False)
        trainable_params['batch_normalize_7he/mean'] = batch_normalize_7he_mean
        batch_normalize_7he_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(conv_5fo.size())[1], dtype=torch_types['float32']))
        batch_normalize_7he_variance = torch.nn.parameter.Parameter(data=batch_normalize_7he_variance_init, requires_grad=False)
        trainable_params['batch_normalize_7he/variance'] = batch_normalize_7he_variance
        batch_normalize_7he_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(conv_5fo.size())[1], dtype=torch_types['float32']))
        batch_normalize_7he_offset = torch.nn.parameter.Parameter(data=batch_normalize_7he_offset_init, requires_grad=True)
        trainable_params['batch_normalize_7he/offset'] = batch_normalize_7he_offset
        batch_normalize_7he_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(conv_5fo.size())[1], dtype=torch_types['float32']))
        batch_normalize_7he_scale = torch.nn.parameter.Parameter(data=batch_normalize_7he_scale_init, requires_grad=True)
        trainable_params['batch_normalize_7he/scale'] = batch_normalize_7he_scale
        batch_normalize_7he = torch.nn.functional.batch_norm(input=conv_5fo, running_mean=batch_normalize_7he_mean, running_var=batch_normalize_7he_variance, weight=batch_normalize_7he_scale, bias=batch_normalize_7he_offset, training=training, momentum=0.1, eps=0.001)
        relu_9ju = torch.nn.functional.relu(input=batch_normalize_7he, inplace=False)
        resnet_16_24yk_conv_11lk_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(relu_9ju.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_16_24yk_conv_11lk_filters = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_11lk_filters_init, requires_grad=True)
        trainable_params['resnet_16_24yk/conv_11lk/filters'] = resnet_16_24yk_conv_11lk_filters
        resnet_16_24yk_conv_11lk = torch.nn.functional.conv2d(input=relu_9ju, weight=resnet_16_24yk_conv_11lk_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_batch_normalize_13na_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_24yk_conv_11lk.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_batch_normalize_13na_mean = torch.nn.parameter.Parameter(data=resnet_16_24yk_batch_normalize_13na_mean_init, requires_grad=False)
        trainable_params['resnet_16_24yk/batch_normalize_13na/mean'] = resnet_16_24yk_batch_normalize_13na_mean
        resnet_16_24yk_batch_normalize_13na_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_24yk_conv_11lk.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_batch_normalize_13na_variance = torch.nn.parameter.Parameter(data=resnet_16_24yk_batch_normalize_13na_variance_init, requires_grad=False)
        trainable_params['resnet_16_24yk/batch_normalize_13na/variance'] = resnet_16_24yk_batch_normalize_13na_variance
        resnet_16_24yk_batch_normalize_13na_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_24yk_conv_11lk.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_batch_normalize_13na_offset = torch.nn.parameter.Parameter(data=resnet_16_24yk_batch_normalize_13na_offset_init, requires_grad=True)
        trainable_params['resnet_16_24yk/batch_normalize_13na/offset'] = resnet_16_24yk_batch_normalize_13na_offset
        resnet_16_24yk_batch_normalize_13na_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_24yk_conv_11lk.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_batch_normalize_13na_scale = torch.nn.parameter.Parameter(data=resnet_16_24yk_batch_normalize_13na_scale_init, requires_grad=True)
        trainable_params['resnet_16_24yk/batch_normalize_13na/scale'] = resnet_16_24yk_batch_normalize_13na_scale
        resnet_16_24yk_batch_normalize_13na = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_11lk, running_mean=resnet_16_24yk_batch_normalize_13na_mean, running_var=resnet_16_24yk_batch_normalize_13na_variance, weight=resnet_16_24yk_batch_normalize_13na_scale, bias=resnet_16_24yk_batch_normalize_13na_offset, training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_relu_15pq = torch.nn.functional.relu(input=resnet_16_24yk_batch_normalize_13na, inplace=False)
        resnet_16_24yk_conv_17rg_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(16, list(resnet_16_24yk_relu_15pq.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_16_24yk_conv_17rg_filters = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_17rg_filters_init, requires_grad=True)
        trainable_params['resnet_16_24yk/conv_17rg/filters'] = resnet_16_24yk_conv_17rg_filters
        resnet_16_24yk_conv_17rg = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_15pq, weight=resnet_16_24yk_conv_17rg_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_conv_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_24yk_conv_17rg.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_conv_mean = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_mean_init, requires_grad=False)
        trainable_params['resnet_16_24yk/conv/mean'] = resnet_16_24yk_conv_mean
        resnet_16_24yk_conv_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_24yk_conv_17rg.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_conv_variance = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_variance_init, requires_grad=False)
        trainable_params['resnet_16_24yk/conv/variance'] = resnet_16_24yk_conv_variance
        resnet_16_24yk_conv_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_16_24yk_conv_17rg.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_conv_offset = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_offset_init, requires_grad=True)
        trainable_params['resnet_16_24yk/conv/offset'] = resnet_16_24yk_conv_offset
        resnet_16_24yk_conv_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_16_24yk_conv_17rg.size())[1], dtype=torch_types['float32']))
        resnet_16_24yk_conv_scale = torch.nn.parameter.Parameter(data=resnet_16_24yk_conv_scale_init, requires_grad=True)
        trainable_params['resnet_16_24yk/conv/scale'] = resnet_16_24yk_conv_scale
        resnet_16_24yk_conv = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_17rg, running_mean=resnet_16_24yk_conv_mean, running_var=resnet_16_24yk_conv_variance, weight=resnet_16_24yk_conv_scale, bias=resnet_16_24yk_conv_offset, training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_add_21vm = torch.add(input=[relu_9ju, resnet_16_24yk_conv][0], other=[relu_9ju, resnet_16_24yk_conv][1])
        resnet_16_24yk_relu_23xc = torch.nn.functional.relu(input=resnet_16_24yk_add_21vm, inplace=False)
        resnet_32_short_cut_43rg_conv0_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(32, list(resnet_16_24yk_relu_23xc.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_conv0_filters = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_conv0_filters_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv0/filters'] = resnet_32_short_cut_43rg_conv0_filters
        resnet_32_short_cut_43rg_conv0 = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=resnet_32_short_cut_43rg_conv0_filters, bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_28cq_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv0.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_28cq_mean = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_28cq_mean_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'] = resnet_32_short_cut_43rg_batch_normalize_28cq_mean
        resnet_32_short_cut_43rg_batch_normalize_28cq_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv0.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_28cq_variance = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_28cq_variance_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'] = resnet_32_short_cut_43rg_batch_normalize_28cq_variance
        resnet_32_short_cut_43rg_batch_normalize_28cq_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv0.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_28cq_offset = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_28cq_offset_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'] = resnet_32_short_cut_43rg_batch_normalize_28cq_offset
        resnet_32_short_cut_43rg_batch_normalize_28cq_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv0.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_28cq_scale = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_28cq_scale_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'] = resnet_32_short_cut_43rg_batch_normalize_28cq_scale
        resnet_32_short_cut_43rg_batch_normalize_28cq = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv0, running_mean=resnet_32_short_cut_43rg_batch_normalize_28cq_mean, running_var=resnet_32_short_cut_43rg_batch_normalize_28cq_variance, weight=resnet_32_short_cut_43rg_batch_normalize_28cq_scale, bias=resnet_32_short_cut_43rg_batch_normalize_28cq_offset, training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_relu_30eg = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_28cq, inplace=False)
        resnet_32_short_cut_43rg_conv_32gw_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(32, list(resnet_32_short_cut_43rg_relu_30eg.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_conv_32gw_filters = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_conv_32gw_filters_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'] = resnet_32_short_cut_43rg_conv_32gw_filters
        resnet_32_short_cut_43rg_conv_32gw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_relu_30eg, weight=resnet_32_short_cut_43rg_conv_32gw_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_34im_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_32gw.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_34im_mean = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_34im_mean_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'] = resnet_32_short_cut_43rg_batch_normalize_34im_mean
        resnet_32_short_cut_43rg_batch_normalize_34im_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_32gw.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_34im_variance = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_34im_variance_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'] = resnet_32_short_cut_43rg_batch_normalize_34im_variance
        resnet_32_short_cut_43rg_batch_normalize_34im_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_32gw.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_34im_offset = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_34im_offset_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'] = resnet_32_short_cut_43rg_batch_normalize_34im_offset
        resnet_32_short_cut_43rg_batch_normalize_34im_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_32gw.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_batch_normalize_34im_scale = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_batch_normalize_34im_scale_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'] = resnet_32_short_cut_43rg_batch_normalize_34im_scale
        resnet_32_short_cut_43rg_batch_normalize_34im = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_32gw, running_mean=resnet_32_short_cut_43rg_batch_normalize_34im_mean, running_var=resnet_32_short_cut_43rg_batch_normalize_34im_variance, weight=resnet_32_short_cut_43rg_batch_normalize_34im_scale, bias=resnet_32_short_cut_43rg_batch_normalize_34im_offset, training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_conv1 = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_34im, inplace=False)
        resnet_32_short_cut_43rg_conv_38ms_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(32, list(resnet_16_24yk_relu_23xc.size())[1], 1, 1, dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_conv_38ms_filters = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_conv_38ms_filters_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'] = resnet_32_short_cut_43rg_conv_38ms_filters
        resnet_32_short_cut_43rg_conv_38ms = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=resnet_32_short_cut_43rg_conv_38ms_filters, bias=None, stride=[2, 2], padding=[0, 0], dilation=1, groups=1)
        resnet_32_short_cut_43rg_short_cut_16_32_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_38ms.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_short_cut_16_32_mean = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_short_cut_16_32_mean_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'] = resnet_32_short_cut_43rg_short_cut_16_32_mean
        resnet_32_short_cut_43rg_short_cut_16_32_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_38ms.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_short_cut_16_32_variance = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_short_cut_16_32_variance_init, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'] = resnet_32_short_cut_43rg_short_cut_16_32_variance
        resnet_32_short_cut_43rg_short_cut_16_32_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_38ms.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_short_cut_16_32_offset = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_short_cut_16_32_offset_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'] = resnet_32_short_cut_43rg_short_cut_16_32_offset
        resnet_32_short_cut_43rg_short_cut_16_32_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_short_cut_43rg_conv_38ms.size())[1], dtype=torch_types['float32']))
        resnet_32_short_cut_43rg_short_cut_16_32_scale = torch.nn.parameter.Parameter(data=resnet_32_short_cut_43rg_short_cut_16_32_scale_init, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'] = resnet_32_short_cut_43rg_short_cut_16_32_scale
        resnet_32_short_cut_43rg_short_cut_16_32 = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_38ms, running_mean=resnet_32_short_cut_43rg_short_cut_16_32_mean, running_var=resnet_32_short_cut_43rg_short_cut_16_32_variance, weight=resnet_32_short_cut_43rg_short_cut_16_32_scale, bias=resnet_32_short_cut_43rg_short_cut_16_32_offset, training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_add_42qy = torch.add(input=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][0], other=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][1])
        resnet_32_58gw_conv_45tw_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(32, list(resnet_32_short_cut_43rg_add_42qy.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_32_58gw_conv_45tw_filters = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_45tw_filters_init, requires_grad=True)
        trainable_params['resnet_32_58gw/conv_45tw/filters'] = resnet_32_58gw_conv_45tw_filters
        resnet_32_58gw_conv_45tw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_add_42qy, weight=resnet_32_58gw_conv_45tw_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_batch_normalize_47vm_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_58gw_conv_45tw.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_batch_normalize_47vm_mean = torch.nn.parameter.Parameter(data=resnet_32_58gw_batch_normalize_47vm_mean_init, requires_grad=False)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'] = resnet_32_58gw_batch_normalize_47vm_mean
        resnet_32_58gw_batch_normalize_47vm_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_58gw_conv_45tw.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_batch_normalize_47vm_variance = torch.nn.parameter.Parameter(data=resnet_32_58gw_batch_normalize_47vm_variance_init, requires_grad=False)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'] = resnet_32_58gw_batch_normalize_47vm_variance
        resnet_32_58gw_batch_normalize_47vm_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_58gw_conv_45tw.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_batch_normalize_47vm_offset = torch.nn.parameter.Parameter(data=resnet_32_58gw_batch_normalize_47vm_offset_init, requires_grad=True)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'] = resnet_32_58gw_batch_normalize_47vm_offset
        resnet_32_58gw_batch_normalize_47vm_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_58gw_conv_45tw.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_batch_normalize_47vm_scale = torch.nn.parameter.Parameter(data=resnet_32_58gw_batch_normalize_47vm_scale_init, requires_grad=True)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'] = resnet_32_58gw_batch_normalize_47vm_scale
        resnet_32_58gw_batch_normalize_47vm = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_45tw, running_mean=resnet_32_58gw_batch_normalize_47vm_mean, running_var=resnet_32_58gw_batch_normalize_47vm_variance, weight=resnet_32_58gw_batch_normalize_47vm_scale, bias=resnet_32_58gw_batch_normalize_47vm_offset, training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_relu_49xc = torch.nn.functional.relu(input=resnet_32_58gw_batch_normalize_47vm, inplace=False)
        resnet_32_58gw_conv_51zs_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(32, list(resnet_32_58gw_relu_49xc.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_32_58gw_conv_51zs_filters = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_51zs_filters_init, requires_grad=True)
        trainable_params['resnet_32_58gw/conv_51zs/filters'] = resnet_32_58gw_conv_51zs_filters
        resnet_32_58gw_conv_51zs = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_49xc, weight=resnet_32_58gw_conv_51zs_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_conv_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_58gw_conv_51zs.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_conv_mean = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_mean_init, requires_grad=False)
        trainable_params['resnet_32_58gw/conv/mean'] = resnet_32_58gw_conv_mean
        resnet_32_58gw_conv_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_58gw_conv_51zs.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_conv_variance = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_variance_init, requires_grad=False)
        trainable_params['resnet_32_58gw/conv/variance'] = resnet_32_58gw_conv_variance
        resnet_32_58gw_conv_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_32_58gw_conv_51zs.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_conv_offset = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_offset_init, requires_grad=True)
        trainable_params['resnet_32_58gw/conv/offset'] = resnet_32_58gw_conv_offset
        resnet_32_58gw_conv_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_32_58gw_conv_51zs.size())[1], dtype=torch_types['float32']))
        resnet_32_58gw_conv_scale = torch.nn.parameter.Parameter(data=resnet_32_58gw_conv_scale_init, requires_grad=True)
        trainable_params['resnet_32_58gw/conv/scale'] = resnet_32_58gw_conv_scale
        resnet_32_58gw_conv = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_51zs, running_mean=resnet_32_58gw_conv_mean, running_var=resnet_32_58gw_conv_variance, weight=resnet_32_58gw_conv_scale, bias=resnet_32_58gw_conv_offset, training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_add_55dy = torch.add(input=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][0], other=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][1])
        resnet_32_58gw_relu_57fo = torch.nn.functional.relu(input=resnet_32_58gw_add_55dy, inplace=False)
        resnet_64_short_cut_77zs_conv0_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(resnet_32_58gw_relu_57fo.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_conv0_filters = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_conv0_filters_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv0/filters'] = resnet_64_short_cut_77zs_conv0_filters
        resnet_64_short_cut_77zs_conv0 = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=resnet_64_short_cut_77zs_conv0_filters, bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_62kc_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv0.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_62kc_mean = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_62kc_mean_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'] = resnet_64_short_cut_77zs_batch_normalize_62kc_mean
        resnet_64_short_cut_77zs_batch_normalize_62kc_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv0.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_62kc_variance = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_62kc_variance_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'] = resnet_64_short_cut_77zs_batch_normalize_62kc_variance
        resnet_64_short_cut_77zs_batch_normalize_62kc_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv0.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_62kc_offset = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_62kc_offset_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'] = resnet_64_short_cut_77zs_batch_normalize_62kc_offset
        resnet_64_short_cut_77zs_batch_normalize_62kc_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv0.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_62kc_scale = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_62kc_scale_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'] = resnet_64_short_cut_77zs_batch_normalize_62kc_scale
        resnet_64_short_cut_77zs_batch_normalize_62kc = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv0, running_mean=resnet_64_short_cut_77zs_batch_normalize_62kc_mean, running_var=resnet_64_short_cut_77zs_batch_normalize_62kc_variance, weight=resnet_64_short_cut_77zs_batch_normalize_62kc_scale, bias=resnet_64_short_cut_77zs_batch_normalize_62kc_offset, training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_relu_64ms = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_62kc, inplace=False)
        resnet_64_short_cut_77zs_conv_66oi_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(resnet_64_short_cut_77zs_relu_64ms.size())[1], 3, 3, dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_conv_66oi_filters = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_conv_66oi_filters_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'] = resnet_64_short_cut_77zs_conv_66oi_filters
        resnet_64_short_cut_77zs_conv_66oi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_relu_64ms, weight=resnet_64_short_cut_77zs_conv_66oi_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_68qy_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_66oi.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_68qy_mean = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_68qy_mean_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'] = resnet_64_short_cut_77zs_batch_normalize_68qy_mean
        resnet_64_short_cut_77zs_batch_normalize_68qy_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_66oi.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_68qy_variance = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_68qy_variance_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'] = resnet_64_short_cut_77zs_batch_normalize_68qy_variance
        resnet_64_short_cut_77zs_batch_normalize_68qy_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_66oi.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_68qy_offset = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_68qy_offset_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'] = resnet_64_short_cut_77zs_batch_normalize_68qy_offset
        resnet_64_short_cut_77zs_batch_normalize_68qy_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_66oi.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_batch_normalize_68qy_scale = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_batch_normalize_68qy_scale_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'] = resnet_64_short_cut_77zs_batch_normalize_68qy_scale
        resnet_64_short_cut_77zs_batch_normalize_68qy = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_66oi, running_mean=resnet_64_short_cut_77zs_batch_normalize_68qy_mean, running_var=resnet_64_short_cut_77zs_batch_normalize_68qy_variance, weight=resnet_64_short_cut_77zs_batch_normalize_68qy_scale, bias=resnet_64_short_cut_77zs_batch_normalize_68qy_offset, training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_conv1 = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_68qy, inplace=False)
        resnet_64_short_cut_77zs_conv_72ue_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(resnet_32_58gw_relu_57fo.size())[1], 1, 1, dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_conv_72ue_filters = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_conv_72ue_filters_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'] = resnet_64_short_cut_77zs_conv_72ue_filters
        resnet_64_short_cut_77zs_conv_72ue = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=resnet_64_short_cut_77zs_conv_72ue_filters, bias=None, stride=[2, 2], padding=[0, 0], dilation=1, groups=1)
        resnet_64_short_cut_77zs_short_cut_32_64_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_72ue.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_short_cut_32_64_mean = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_short_cut_32_64_mean_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'] = resnet_64_short_cut_77zs_short_cut_32_64_mean
        resnet_64_short_cut_77zs_short_cut_32_64_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_72ue.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_short_cut_32_64_variance = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_short_cut_32_64_variance_init, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'] = resnet_64_short_cut_77zs_short_cut_32_64_variance
        resnet_64_short_cut_77zs_short_cut_32_64_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_72ue.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_short_cut_32_64_offset = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_short_cut_32_64_offset_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'] = resnet_64_short_cut_77zs_short_cut_32_64_offset
        resnet_64_short_cut_77zs_short_cut_32_64_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(resnet_64_short_cut_77zs_conv_72ue.size())[1], dtype=torch_types['float32']))
        resnet_64_short_cut_77zs_short_cut_32_64_scale = torch.nn.parameter.Parameter(data=resnet_64_short_cut_77zs_short_cut_32_64_scale_init, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'] = resnet_64_short_cut_77zs_short_cut_32_64_scale
        resnet_64_short_cut_77zs_short_cut_32_64 = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_72ue, running_mean=resnet_64_short_cut_77zs_short_cut_32_64_mean, running_var=resnet_64_short_cut_77zs_short_cut_32_64_variance, weight=resnet_64_short_cut_77zs_short_cut_32_64_scale, bias=resnet_64_short_cut_77zs_short_cut_32_64_offset, training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_add_76yk = torch.add(input=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][0], other=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][1])
        features_conv_79bi_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(resnet_64_short_cut_77zs_add_76yk.size())[1], 3, 3, dtype=torch_types['float32']))
        features_conv_79bi_filters = torch.nn.parameter.Parameter(data=features_conv_79bi_filters_init, requires_grad=True)
        trainable_params['features/conv_79bi/filters'] = features_conv_79bi_filters
        features_conv_79bi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_add_76yk, weight=features_conv_79bi_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        features_batch_normalize_81dy_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(features_conv_79bi.size())[1], dtype=torch_types['float32']))
        features_batch_normalize_81dy_mean = torch.nn.parameter.Parameter(data=features_batch_normalize_81dy_mean_init, requires_grad=False)
        trainable_params['features/batch_normalize_81dy/mean'] = features_batch_normalize_81dy_mean
        features_batch_normalize_81dy_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(features_conv_79bi.size())[1], dtype=torch_types['float32']))
        features_batch_normalize_81dy_variance = torch.nn.parameter.Parameter(data=features_batch_normalize_81dy_variance_init, requires_grad=False)
        trainable_params['features/batch_normalize_81dy/variance'] = features_batch_normalize_81dy_variance
        features_batch_normalize_81dy_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(features_conv_79bi.size())[1], dtype=torch_types['float32']))
        features_batch_normalize_81dy_offset = torch.nn.parameter.Parameter(data=features_batch_normalize_81dy_offset_init, requires_grad=True)
        trainable_params['features/batch_normalize_81dy/offset'] = features_batch_normalize_81dy_offset
        features_batch_normalize_81dy_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(features_conv_79bi.size())[1], dtype=torch_types['float32']))
        features_batch_normalize_81dy_scale = torch.nn.parameter.Parameter(data=features_batch_normalize_81dy_scale_init, requires_grad=True)
        trainable_params['features/batch_normalize_81dy/scale'] = features_batch_normalize_81dy_scale
        features_batch_normalize_81dy = torch.nn.functional.batch_norm(input=features_conv_79bi, running_mean=features_batch_normalize_81dy_mean, running_var=features_batch_normalize_81dy_variance, weight=features_batch_normalize_81dy_scale, bias=features_batch_normalize_81dy_offset, training=training, momentum=0.1, eps=0.001)
        features_relu_83fo = torch.nn.functional.relu(input=features_batch_normalize_81dy, inplace=False)
        features_conv_85he_filters_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(64, list(features_relu_83fo.size())[1], 3, 3, dtype=torch_types['float32']))
        features_conv_85he_filters = torch.nn.parameter.Parameter(data=features_conv_85he_filters_init, requires_grad=True)
        trainable_params['features/conv_85he/filters'] = features_conv_85he_filters
        features_conv_85he = torch.nn.functional.conv2d(input=features_relu_83fo, weight=features_conv_85he_filters, bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        features_conv_mean_init = torch.nn.init.zeros_(tensor=torch.empty(list(features_conv_85he.size())[1], dtype=torch_types['float32']))
        features_conv_mean = torch.nn.parameter.Parameter(data=features_conv_mean_init, requires_grad=False)
        trainable_params['features/conv/mean'] = features_conv_mean
        features_conv_variance_init = torch.nn.init.ones_(tensor=torch.empty(list(features_conv_85he.size())[1], dtype=torch_types['float32']))
        features_conv_variance = torch.nn.parameter.Parameter(data=features_conv_variance_init, requires_grad=False)
        trainable_params['features/conv/variance'] = features_conv_variance
        features_conv_offset_init = torch.nn.init.zeros_(tensor=torch.empty(list(features_conv_85he.size())[1], dtype=torch_types['float32']))
        features_conv_offset = torch.nn.parameter.Parameter(data=features_conv_offset_init, requires_grad=True)
        trainable_params['features/conv/offset'] = features_conv_offset
        features_conv_scale_init = torch.nn.init.ones_(tensor=torch.empty(list(features_conv_85he.size())[1], dtype=torch_types['float32']))
        features_conv_scale = torch.nn.parameter.Parameter(data=features_conv_scale_init, requires_grad=True)
        trainable_params['features/conv/scale'] = features_conv_scale
        features_conv = torch.nn.functional.batch_norm(input=features_conv_85he, running_mean=features_conv_mean, running_var=features_conv_variance, weight=features_conv_scale, bias=features_conv_offset, training=training, momentum=0.1, eps=0.001)
        features_add_89lk = torch.add(input=[resnet_64_short_cut_77zs_add_76yk, features_conv][0], other=[resnet_64_short_cut_77zs_add_76yk, features_conv][1])
        features_relu_91na = torch.nn.functional.relu(input=features_add_89lk, inplace=False)
        max_pool2d_94qy = torch.nn.functional.max_pool2d(input=features_relu_91na, kernel_size=[3, 3], stride=[2, 2], padding=0)
        flatten_96so = torch.flatten(input=max_pool2d_94qy, start_dim=1, end_dim=-1)
        dense_98ue_weights_init = torch.nn.init.xavier_uniform_(tensor=torch.empty(10, list(flatten_96so.size())[1], dtype=torch_types['float32']))
        dense_98ue_weights = torch.nn.parameter.Parameter(data=dense_98ue_weights_init, requires_grad=True)
        trainable_params['dense_98ue/weights'] = dense_98ue_weights
        dense_98ue_bias_init = torch.nn.init.zeros_(tensor=torch.empty(1, dtype=torch_types['float32']))
        dense_98ue_bias = torch.nn.parameter.Parameter(data=dense_98ue_bias_init, requires_grad=True)
        trainable_params['dense_98ue/bias'] = dense_98ue_bias
        return trainable_params
    
    @staticmethod
    def model(input_data, trainable_params, training):
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=trainable_params['conv_5fo/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        batch_normalize_7he = torch.nn.functional.batch_norm(input=conv_5fo, running_mean=trainable_params['batch_normalize_7he/mean'], running_var=trainable_params['batch_normalize_7he/variance'], weight=trainable_params['batch_normalize_7he/scale'], bias=trainable_params['batch_normalize_7he/offset'], training=training, momentum=0.1, eps=0.001)
        relu_9ju = torch.nn.functional.relu(input=batch_normalize_7he, inplace=False)
        resnet_16_24yk_conv_11lk = torch.nn.functional.conv2d(input=relu_9ju, weight=trainable_params['resnet_16_24yk/conv_11lk/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_batch_normalize_13na = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_11lk, running_mean=trainable_params['resnet_16_24yk/batch_normalize_13na/mean'], running_var=trainable_params['resnet_16_24yk/batch_normalize_13na/variance'], weight=trainable_params['resnet_16_24yk/batch_normalize_13na/scale'], bias=trainable_params['resnet_16_24yk/batch_normalize_13na/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_relu_15pq = torch.nn.functional.relu(input=resnet_16_24yk_batch_normalize_13na, inplace=False)
        resnet_16_24yk_conv_17rg = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_15pq, weight=trainable_params['resnet_16_24yk/conv_17rg/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_conv = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_17rg, running_mean=trainable_params['resnet_16_24yk/conv/mean'], running_var=trainable_params['resnet_16_24yk/conv/variance'], weight=trainable_params['resnet_16_24yk/conv/scale'], bias=trainable_params['resnet_16_24yk/conv/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_add_21vm = torch.add(input=[relu_9ju, resnet_16_24yk_conv][0], other=[relu_9ju, resnet_16_24yk_conv][1])
        resnet_16_24yk_relu_23xc = torch.nn.functional.relu(input=resnet_16_24yk_add_21vm, inplace=False)
        resnet_32_short_cut_43rg_conv0 = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=trainable_params['resnet_32_short_cut_43rg/conv0/filters'], bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_28cq = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv0, running_mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'], weight=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'], bias=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_relu_30eg = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_28cq, inplace=False)
        resnet_32_short_cut_43rg_conv_32gw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_relu_30eg, weight=trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_34im = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_32gw, running_mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'], weight=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'], bias=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_conv1 = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_34im, inplace=False)
        resnet_32_short_cut_43rg_conv_38ms = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'], bias=None, stride=[2, 2], padding=[0, 0], dilation=1, groups=1)
        resnet_32_short_cut_43rg_short_cut_16_32 = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_38ms, running_mean=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'], weight=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'], bias=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_add_42qy = torch.add(input=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][0], other=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][1])
        resnet_32_58gw_conv_45tw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_add_42qy, weight=trainable_params['resnet_32_58gw/conv_45tw/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_batch_normalize_47vm = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_45tw, running_mean=trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'], running_var=trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'], weight=trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'], bias=trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_relu_49xc = torch.nn.functional.relu(input=resnet_32_58gw_batch_normalize_47vm, inplace=False)
        resnet_32_58gw_conv_51zs = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_49xc, weight=trainable_params['resnet_32_58gw/conv_51zs/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_conv = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_51zs, running_mean=trainable_params['resnet_32_58gw/conv/mean'], running_var=trainable_params['resnet_32_58gw/conv/variance'], weight=trainable_params['resnet_32_58gw/conv/scale'], bias=trainable_params['resnet_32_58gw/conv/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_add_55dy = torch.add(input=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][0], other=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][1])
        resnet_32_58gw_relu_57fo = torch.nn.functional.relu(input=resnet_32_58gw_add_55dy, inplace=False)
        resnet_64_short_cut_77zs_conv0 = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=trainable_params['resnet_64_short_cut_77zs/conv0/filters'], bias=None, stride=[2, 2], padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_62kc = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv0, running_mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'], weight=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'], bias=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_relu_64ms = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_62kc, inplace=False)
        resnet_64_short_cut_77zs_conv_66oi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_relu_64ms, weight=trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_68qy = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_66oi, running_mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'], weight=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'], bias=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_conv1 = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_68qy, inplace=False)
        resnet_64_short_cut_77zs_conv_72ue = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'], bias=None, stride=[2, 2], padding=[0, 0], dilation=1, groups=1)
        resnet_64_short_cut_77zs_short_cut_32_64 = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_72ue, running_mean=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'], weight=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'], bias=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_add_76yk = torch.add(input=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][0], other=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][1])
        features_conv_79bi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_add_76yk, weight=trainable_params['features/conv_79bi/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        features_batch_normalize_81dy = torch.nn.functional.batch_norm(input=features_conv_79bi, running_mean=trainable_params['features/batch_normalize_81dy/mean'], running_var=trainable_params['features/batch_normalize_81dy/variance'], weight=trainable_params['features/batch_normalize_81dy/scale'], bias=trainable_params['features/batch_normalize_81dy/offset'], training=training, momentum=0.1, eps=0.001)
        features_relu_83fo = torch.nn.functional.relu(input=features_batch_normalize_81dy, inplace=False)
        features_conv_85he = torch.nn.functional.conv2d(input=features_relu_83fo, weight=trainable_params['features/conv_85he/filters'], bias=None, stride=[1, 1], padding=[1, 1], dilation=1, groups=1)
        features_conv = torch.nn.functional.batch_norm(input=features_conv_85he, running_mean=trainable_params['features/conv/mean'], running_var=trainable_params['features/conv/variance'], weight=trainable_params['features/conv/scale'], bias=trainable_params['features/conv/offset'], training=training, momentum=0.1, eps=0.001)
        features_add_89lk = torch.add(input=[resnet_64_short_cut_77zs_add_76yk, features_conv][0], other=[resnet_64_short_cut_77zs_add_76yk, features_conv][1])
        features_relu_91na = torch.nn.functional.relu(input=features_add_89lk, inplace=False)
        max_pool2d_94qy = torch.nn.functional.max_pool2d(input=features_relu_91na, kernel_size=[3, 3], stride=[2, 2], padding=0)
        flatten_96so = torch.flatten(input=max_pool2d_94qy, start_dim=1, end_dim=-1)
        dense_98ue = torch.nn.functional.linear(input=flatten_96so, weight=trainable_params['dense_98ue/weights'], bias=trainable_params['dense_98ue/bias'])
        d_1 = torch.nn.functional.dropout(input=dense_98ue, p=0.2, training=training, inplace=False)
        return d_1 
    
    @staticmethod
    def get_loss(d_1, labels, trainable_params):
        cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=[labels, d_1][0], input=[labels, d_1][1])
        regularizer1 = 0.002*sum(list(map(lambda x: torch.norm(trainable_params[x]), ['conv_5fo/filters', 'resnet_16_24yk/conv_11lk/filters', 'resnet_16_24yk/conv_17rg/filters', 'resnet_32_short_cut_43rg/conv0/filters', 'resnet_32_short_cut_43rg/conv_32gw/filters', 'resnet_32_short_cut_43rg/conv_38ms/filters', 'resnet_32_58gw/conv_45tw/filters', 'resnet_32_58gw/conv_51zs/filters', 'resnet_64_short_cut_77zs/conv0/filters', 'resnet_64_short_cut_77zs/conv_66oi/filters', 'resnet_64_short_cut_77zs/conv_72ue/filters', 'features/conv_79bi/filters', 'features/conv_85he/filters'])))
        losses = torch.add(input=[cross_0, regularizer1][0], other=[cross_0, regularizer1][1])
        return losses 
    
    @staticmethod
    def get_optimizer(trainable_params):
        solver = {'optimizer': torch.optim.Adam(params=trainable_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08), 'learning_rate': lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1, verbose=False)}
        return solver 
    