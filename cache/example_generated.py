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
    def get_trainable_params(ckpt):
        trainable_params = dict()
        xavier_uniform_1b342964_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        conv_5fo_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342964_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['conv_5fo/filters'] = conv_5fo_filters
        zeros_initializer_1b34297e_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        batch_normalize_7he_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b34297e_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['batch_normalize_7he/mean'] = batch_normalize_7he_mean
        zeros_initializer_1b342988_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        batch_normalize_7he_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342988_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['batch_normalize_7he/offset'] = batch_normalize_7he_offset
        ones_initializer_1b342990_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        batch_normalize_7he_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342990_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['batch_normalize_7he/scale'] = batch_normalize_7he_scale
        ones_initializer_1b342998_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        batch_normalize_7he_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342998_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['batch_normalize_7he/variance'] = batch_normalize_7he_variance
        xavier_uniform_1b3429a2_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        resnet_16_24yk_conv_11lk_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b3429a2_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/conv_11lk/filters'] = resnet_16_24yk_conv_11lk_filters
        zeros_initializer_1b3429bc_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_batch_normalize_13na_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b3429bc_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_16_24yk/batch_normalize_13na/mean'] = resnet_16_24yk_batch_normalize_13na_mean
        zeros_initializer_1b3429c6_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_batch_normalize_13na_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b3429c6_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/batch_normalize_13na/offset'] = resnet_16_24yk_batch_normalize_13na_offset
        ones_initializer_1b3429ce_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_batch_normalize_13na_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b3429ce_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/batch_normalize_13na/scale'] = resnet_16_24yk_batch_normalize_13na_scale
        ones_initializer_1b3429d6_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_batch_normalize_13na_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b3429d6_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_16_24yk/batch_normalize_13na/variance'] = resnet_16_24yk_batch_normalize_13na_variance
        xavier_uniform_1b3429e0_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        resnet_16_24yk_conv_17rg_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b3429e0_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/conv_17rg/filters'] = resnet_16_24yk_conv_17rg_filters
        zeros_initializer_1b3429fa_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_conv_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b3429fa_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_16_24yk/conv/mean'] = resnet_16_24yk_conv_mean
        zeros_initializer_1b342a04_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_conv_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342a04_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/conv/offset'] = resnet_16_24yk_conv_offset
        ones_initializer_1b342a0c_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_conv_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342a0c_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_16_24yk/conv/scale'] = resnet_16_24yk_conv_scale
        ones_initializer_1b342a14_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        resnet_16_24yk_conv_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342a14_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_16_24yk/conv/variance'] = resnet_16_24yk_conv_variance
        xavier_uniform_1b342a1e_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[32, 16, 3, 3]))
        resnet_32_short_cut_43rg_conv0_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342a1e_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv0/filters'] = resnet_32_short_cut_43rg_conv0_filters
        zeros_initializer_1b342a38_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_28cq_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342a38_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'] = resnet_32_short_cut_43rg_batch_normalize_28cq_mean
        zeros_initializer_1b342a42_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_28cq_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342a42_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'] = resnet_32_short_cut_43rg_batch_normalize_28cq_offset
        ones_initializer_1b342a4a_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_28cq_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342a4a_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'] = resnet_32_short_cut_43rg_batch_normalize_28cq_scale
        ones_initializer_1b342a52_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_28cq_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342a52_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'] = resnet_32_short_cut_43rg_batch_normalize_28cq_variance
        xavier_uniform_1b342a5c_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[32, 32, 3, 3]))
        resnet_32_short_cut_43rg_conv_32gw_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342a5c_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'] = resnet_32_short_cut_43rg_conv_32gw_filters
        zeros_initializer_1b342a76_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_34im_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342a76_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'] = resnet_32_short_cut_43rg_batch_normalize_34im_mean
        zeros_initializer_1b342a80_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_34im_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342a80_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'] = resnet_32_short_cut_43rg_batch_normalize_34im_offset
        ones_initializer_1b342a88_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_34im_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342a88_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'] = resnet_32_short_cut_43rg_batch_normalize_34im_scale
        ones_initializer_1b342a90_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_batch_normalize_34im_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342a90_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'] = resnet_32_short_cut_43rg_batch_normalize_34im_variance
        xavier_uniform_1b342a9a_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[32, 16, 1, 1]))
        resnet_32_short_cut_43rg_conv_38ms_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342a9a_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'] = resnet_32_short_cut_43rg_conv_38ms_filters
        zeros_initializer_1b342ab7_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_short_cut_16_32_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342ab7_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'] = resnet_32_short_cut_43rg_short_cut_16_32_mean
        zeros_initializer_1b342ac1_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_short_cut_16_32_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342ac1_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'] = resnet_32_short_cut_43rg_short_cut_16_32_offset
        ones_initializer_1b342ac9_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_short_cut_16_32_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342ac9_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'] = resnet_32_short_cut_43rg_short_cut_16_32_scale
        ones_initializer_1b342ad1_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_short_cut_43rg_short_cut_16_32_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342ad1_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'] = resnet_32_short_cut_43rg_short_cut_16_32_variance
        xavier_uniform_1b342adb_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[32, 32, 3, 3]))
        resnet_32_58gw_conv_45tw_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342adb_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/conv_45tw/filters'] = resnet_32_58gw_conv_45tw_filters
        zeros_initializer_1b342af5_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_batch_normalize_47vm_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342af5_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'] = resnet_32_58gw_batch_normalize_47vm_mean
        zeros_initializer_1b342aff_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_batch_normalize_47vm_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342aff_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'] = resnet_32_58gw_batch_normalize_47vm_offset
        ones_initializer_1b342b07_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_batch_normalize_47vm_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342b07_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'] = resnet_32_58gw_batch_normalize_47vm_scale
        ones_initializer_1b342b0f_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_batch_normalize_47vm_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342b0f_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'] = resnet_32_58gw_batch_normalize_47vm_variance
        xavier_uniform_1b342b19_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[32, 32, 3, 3]))
        resnet_32_58gw_conv_51zs_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342b19_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/conv_51zs/filters'] = resnet_32_58gw_conv_51zs_filters
        zeros_initializer_1b342b33_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_conv_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342b33_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_58gw/conv/mean'] = resnet_32_58gw_conv_mean
        zeros_initializer_1b342b3d_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_conv_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342b3d_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/conv/offset'] = resnet_32_58gw_conv_offset
        ones_initializer_1b342b45_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_conv_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342b45_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_32_58gw/conv/scale'] = resnet_32_58gw_conv_scale
        ones_initializer_1b342b4d_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[32, ]))
        resnet_32_58gw_conv_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342b4d_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_32_58gw/conv/variance'] = resnet_32_58gw_conv_variance
        xavier_uniform_1b342b57_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 32, 3, 3]))
        resnet_64_short_cut_77zs_conv0_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342b57_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv0/filters'] = resnet_64_short_cut_77zs_conv0_filters
        zeros_initializer_1b342b71_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_62kc_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342b71_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'] = resnet_64_short_cut_77zs_batch_normalize_62kc_mean
        zeros_initializer_1b342b7b_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_62kc_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342b7b_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'] = resnet_64_short_cut_77zs_batch_normalize_62kc_offset
        ones_initializer_1b342b83_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_62kc_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342b83_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'] = resnet_64_short_cut_77zs_batch_normalize_62kc_scale
        ones_initializer_1b342b8b_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_62kc_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342b8b_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'] = resnet_64_short_cut_77zs_batch_normalize_62kc_variance
        xavier_uniform_1b342b95_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 64, 3, 3]))
        resnet_64_short_cut_77zs_conv_66oi_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342b95_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'] = resnet_64_short_cut_77zs_conv_66oi_filters
        zeros_initializer_1b342baf_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_68qy_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342baf_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'] = resnet_64_short_cut_77zs_batch_normalize_68qy_mean
        zeros_initializer_1b342bb9_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_68qy_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342bb9_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'] = resnet_64_short_cut_77zs_batch_normalize_68qy_offset
        ones_initializer_1b342bc1_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_68qy_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342bc1_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'] = resnet_64_short_cut_77zs_batch_normalize_68qy_scale
        ones_initializer_1b342bc9_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_batch_normalize_68qy_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342bc9_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'] = resnet_64_short_cut_77zs_batch_normalize_68qy_variance
        xavier_uniform_1b342bd3_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 32, 1, 1]))
        resnet_64_short_cut_77zs_conv_72ue_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342bd3_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'] = resnet_64_short_cut_77zs_conv_72ue_filters
        zeros_initializer_1b342bed_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_short_cut_32_64_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342bed_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'] = resnet_64_short_cut_77zs_short_cut_32_64_mean
        zeros_initializer_1b342bf7_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_short_cut_32_64_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342bf7_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'] = resnet_64_short_cut_77zs_short_cut_32_64_offset
        ones_initializer_1b342bff_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_short_cut_32_64_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342bff_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'] = resnet_64_short_cut_77zs_short_cut_32_64_scale
        ones_initializer_1b342c07_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        resnet_64_short_cut_77zs_short_cut_32_64_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342c07_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'] = resnet_64_short_cut_77zs_short_cut_32_64_variance
        xavier_uniform_1b342c11_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 64, 3, 3]))
        features_conv_79bi_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342c11_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/conv_79bi/filters'] = features_conv_79bi_filters
        zeros_initializer_1b342c2b_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        features_batch_normalize_81dy_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342c2b_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['features/batch_normalize_81dy/mean'] = features_batch_normalize_81dy_mean
        zeros_initializer_1b342c35_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        features_batch_normalize_81dy_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342c35_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/batch_normalize_81dy/offset'] = features_batch_normalize_81dy_offset
        ones_initializer_1b342c3d_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        features_batch_normalize_81dy_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342c3d_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/batch_normalize_81dy/scale'] = features_batch_normalize_81dy_scale
        ones_initializer_1b342c45_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        features_batch_normalize_81dy_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342c45_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['features/batch_normalize_81dy/variance'] = features_batch_normalize_81dy_variance
        xavier_uniform_1b342c4f_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 64, 3, 3]))
        features_conv_85he_filters = torch.nn.parameter.Parameter(data=xavier_uniform_1b342c4f_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/conv_85he/filters'] = features_conv_85he_filters
        zeros_initializer_1b342c69_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        features_conv_mean = torch.nn.parameter.Parameter(data=zeros_initializer_1b342c69_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['features/conv/mean'] = features_conv_mean
        zeros_initializer_1b342c73_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        features_conv_offset = torch.nn.parameter.Parameter(data=zeros_initializer_1b342c73_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/conv/offset'] = features_conv_offset
        ones_initializer_1b342c7b_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        features_conv_scale = torch.nn.parameter.Parameter(data=ones_initializer_1b342c7b_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['features/conv/scale'] = features_conv_scale
        ones_initializer_1b342c83_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        features_conv_variance = torch.nn.parameter.Parameter(data=ones_initializer_1b342c83_cfc8_11eb_a40f_711d29eed077, requires_grad=False)
        trainable_params['features/conv/variance'] = features_conv_variance
        zeros_initializer_1b342c95_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.zeros_(tensor=torch.empty(*[1, ]))
        dense_98ue_bias = torch.nn.parameter.Parameter(data=zeros_initializer_1b342c95_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['dense_98ue/bias'] = dense_98ue_bias
        xavier_uniform_1b342c9d_cfc8_11eb_a40f_711d29eed077 = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[10, 576]))
        dense_98ue_weights = torch.nn.parameter.Parameter(data=xavier_uniform_1b342c9d_cfc8_11eb_a40f_711d29eed077, requires_grad=True)
        trainable_params['dense_98ue/weights'] = dense_98ue_weights
        return trainable_params
    
    @staticmethod
    def model(input_data, trainable_params, training):
        conv_5fo = torch.nn.functional.conv2d(input=input_data, weight=trainable_params['conv_5fo/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        batch_normalize_7he = torch.nn.functional.batch_norm(input=conv_5fo, running_mean=trainable_params['batch_normalize_7he/mean'], running_var=trainable_params['batch_normalize_7he/variance'], weight=trainable_params['batch_normalize_7he/scale'], bias=trainable_params['batch_normalize_7he/offset'], training=training, momentum=0.1, eps=0.001)
        relu_9ju = torch.nn.functional.relu(input=batch_normalize_7he, inplace=False)
        resnet_16_24yk_conv_11lk = torch.nn.functional.conv2d(input=relu_9ju, weight=trainable_params['resnet_16_24yk/conv_11lk/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_batch_normalize_13na = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_11lk, running_mean=trainable_params['resnet_16_24yk/batch_normalize_13na/mean'], running_var=trainable_params['resnet_16_24yk/batch_normalize_13na/variance'], weight=trainable_params['resnet_16_24yk/batch_normalize_13na/scale'], bias=trainable_params['resnet_16_24yk/batch_normalize_13na/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_relu_15pq = torch.nn.functional.relu(input=resnet_16_24yk_batch_normalize_13na, inplace=False)
        resnet_16_24yk_conv_17rg = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_15pq, weight=trainable_params['resnet_16_24yk/conv_17rg/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_16_24yk_conv = torch.nn.functional.batch_norm(input=resnet_16_24yk_conv_17rg, running_mean=trainable_params['resnet_16_24yk/conv/mean'], running_var=trainable_params['resnet_16_24yk/conv/variance'], weight=trainable_params['resnet_16_24yk/conv/scale'], bias=trainable_params['resnet_16_24yk/conv/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_16_24yk_add_21vm = torch.add(input=[relu_9ju, resnet_16_24yk_conv][0], other=[relu_9ju, resnet_16_24yk_conv][1])
        resnet_16_24yk_relu_23xc = torch.nn.functional.relu(input=resnet_16_24yk_add_21vm, inplace=False)
        resnet_32_short_cut_43rg_conv0 = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=trainable_params['resnet_32_short_cut_43rg/conv0/filters'], bias=None, stride=2, padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_28cq = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv0, running_mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'], weight=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'], bias=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_relu_30eg = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_28cq, inplace=False)
        resnet_32_short_cut_43rg_conv_32gw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_relu_30eg, weight=trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_32_short_cut_43rg_batch_normalize_34im = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_32gw, running_mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'], weight=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'], bias=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_conv1 = torch.nn.functional.relu(input=resnet_32_short_cut_43rg_batch_normalize_34im, inplace=False)
        resnet_32_short_cut_43rg_conv_38ms = torch.nn.functional.conv2d(input=resnet_16_24yk_relu_23xc, weight=trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'], bias=None, stride=2, padding=[0, 0], dilation=1, groups=1)
        resnet_32_short_cut_43rg_short_cut_16_32 = torch.nn.functional.batch_norm(input=resnet_32_short_cut_43rg_conv_38ms, running_mean=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'], running_var=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'], weight=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'], bias=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_short_cut_43rg_add_42qy = torch.add(input=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][0], other=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][1])
        resnet_32_58gw_conv_45tw = torch.nn.functional.conv2d(input=resnet_32_short_cut_43rg_add_42qy, weight=trainable_params['resnet_32_58gw/conv_45tw/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_batch_normalize_47vm = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_45tw, running_mean=trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'], running_var=trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'], weight=trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'], bias=trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_relu_49xc = torch.nn.functional.relu(input=resnet_32_58gw_batch_normalize_47vm, inplace=False)
        resnet_32_58gw_conv_51zs = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_49xc, weight=trainable_params['resnet_32_58gw/conv_51zs/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_32_58gw_conv = torch.nn.functional.batch_norm(input=resnet_32_58gw_conv_51zs, running_mean=trainable_params['resnet_32_58gw/conv/mean'], running_var=trainable_params['resnet_32_58gw/conv/variance'], weight=trainable_params['resnet_32_58gw/conv/scale'], bias=trainable_params['resnet_32_58gw/conv/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_32_58gw_add_55dy = torch.add(input=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][0], other=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][1])
        resnet_32_58gw_relu_57fo = torch.nn.functional.relu(input=resnet_32_58gw_add_55dy, inplace=False)
        resnet_64_short_cut_77zs_conv0 = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=trainable_params['resnet_64_short_cut_77zs/conv0/filters'], bias=None, stride=2, padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_62kc = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv0, running_mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'], weight=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'], bias=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_relu_64ms = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_62kc, inplace=False)
        resnet_64_short_cut_77zs_conv_66oi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_relu_64ms, weight=trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        resnet_64_short_cut_77zs_batch_normalize_68qy = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_66oi, running_mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'], weight=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'], bias=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_conv1 = torch.nn.functional.relu(input=resnet_64_short_cut_77zs_batch_normalize_68qy, inplace=False)
        resnet_64_short_cut_77zs_conv_72ue = torch.nn.functional.conv2d(input=resnet_32_58gw_relu_57fo, weight=trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'], bias=None, stride=2, padding=[0, 0], dilation=1, groups=1)
        resnet_64_short_cut_77zs_short_cut_32_64 = torch.nn.functional.batch_norm(input=resnet_64_short_cut_77zs_conv_72ue, running_mean=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'], running_var=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'], weight=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'], bias=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'], training=training, momentum=0.1, eps=0.001)
        resnet_64_short_cut_77zs_add_76yk = torch.add(input=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][0], other=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][1])
        features_conv_79bi = torch.nn.functional.conv2d(input=resnet_64_short_cut_77zs_add_76yk, weight=trainable_params['features/conv_79bi/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        features_batch_normalize_81dy = torch.nn.functional.batch_norm(input=features_conv_79bi, running_mean=trainable_params['features/batch_normalize_81dy/mean'], running_var=trainable_params['features/batch_normalize_81dy/variance'], weight=trainable_params['features/batch_normalize_81dy/scale'], bias=trainable_params['features/batch_normalize_81dy/offset'], training=training, momentum=0.1, eps=0.001)
        features_relu_83fo = torch.nn.functional.relu(input=features_batch_normalize_81dy, inplace=False)
        features_conv_85he = torch.nn.functional.conv2d(input=features_relu_83fo, weight=trainable_params['features/conv_85he/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        features_conv = torch.nn.functional.batch_norm(input=features_conv_85he, running_mean=trainable_params['features/conv/mean'], running_var=trainable_params['features/conv/variance'], weight=trainable_params['features/conv/scale'], bias=trainable_params['features/conv/offset'], training=training, momentum=0.1, eps=0.001)
        features_add_89lk = torch.add(input=[resnet_64_short_cut_77zs_add_76yk, features_conv][0], other=[resnet_64_short_cut_77zs_add_76yk, features_conv][1])
        features_relu_91na = torch.nn.functional.relu(input=features_add_89lk, inplace=False)
        max_pool2d_94qy = torch.nn.functional.max_pool2d(input=features_relu_91na, kernel_size=3, stride=2, padding=[0, 0])
        flatten_96so = torch.flatten(input=max_pool2d_94qy, start_dim=1, end_dim=-1)
        dense_98ue = torch.nn.functional.linear(weight=trainable_params['dense_98ue/weights'], bias=trainable_params['dense_98ue/bias'], input=flatten_96so)
        d_1 = torch.nn.functional.dropout(input=dense_98ue, p=0.2, training=training, inplace=False)
        return d_1 
    
    @staticmethod
    def get_loss(trainable_params, inputs):
        cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=inputs[0], input=inputs[1])
        regularizer1 = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['conv_5fo/filters', 'resnet_16_24yk/conv_11lk/filters', 'resnet_16_24yk/conv_17rg/filters', 'resnet_32_short_cut_43rg/conv0/filters', 'resnet_32_short_cut_43rg/conv_32gw/filters', 'resnet_32_short_cut_43rg/conv_38ms/filters', 'resnet_32_58gw/conv_45tw/filters', 'resnet_32_58gw/conv_51zs/filters', 'resnet_64_short_cut_77zs/conv0/filters', 'resnet_64_short_cut_77zs/conv_66oi/filters', 'resnet_64_short_cut_77zs/conv_72ue/filters', 'features/conv_79bi/filters', 'features/conv_85he/filters'])))
        losses = torch.add(input=[cross_0, regularizer1][0], other=[cross_0, regularizer1][1])
        return losses 
    
    @staticmethod
    def get_optimizer(trainable_params):
        exponential_decay_1b342cae_cfc8_11eb_a40f_711d29eed077 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1, verbose=False)
        solver = torch.optim.Adam(params=trainable_params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        return solver 
    

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
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
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small1.yml",
               ["cache",  "config_1622420349826577.json"],
               ["checkpoints", None])

ckpt = C.load()

model = Model(ckpt)

model.to(device)

optimizer = model.get_optimizer(model.params)

learning_rate = model.get_scheduler(optimizer)


for epoch in range(90):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs, True)
        loss = model.get_loss(model.trainable_params, [labels, output])
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            C.save(model.trainable_params)
            #outputs = model(inputs, trainable_params, False)
            #_loss = get_loss( outputs, labels, trainable_params)
            running_loss = 0.0
    learning_rate.step()
print('Finished Training')


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images, False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


