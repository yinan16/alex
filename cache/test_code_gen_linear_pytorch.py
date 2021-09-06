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

    def forward(self, data_block_input_data, probes, trainable_params, training):
        x = self.model(data_block_input_data, probes, trainable_params, training)
        return x

    @staticmethod
    def get_trainable_params(ckpt):
        trainable_params = dict()
        model_block_conv_6gw_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 3, 3, 3]))
        model_block_conv_6gw_filters = torch.nn.parameter.Parameter(data=model_block_conv_6gw_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
        model_block_batch_normalize_10kc_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        model_block_batch_normalize_10kc_mean = torch.nn.parameter.Parameter(data=model_block_batch_normalize_10kc_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_10kc/mean'] = model_block_batch_normalize_10kc_mean
        model_block_batch_normalize_10kc_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[64, ]))
        model_block_batch_normalize_10kc_offset = torch.nn.parameter.Parameter(data=model_block_batch_normalize_10kc_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_10kc/offset'] = model_block_batch_normalize_10kc_offset
        model_block_batch_normalize_10kc_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        model_block_batch_normalize_10kc_scale = torch.nn.parameter.Parameter(data=model_block_batch_normalize_10kc_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_10kc/scale'] = model_block_batch_normalize_10kc_scale
        model_block_batch_normalize_10kc_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[64, ]))
        model_block_batch_normalize_10kc_variance = torch.nn.parameter.Parameter(data=model_block_batch_normalize_10kc_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_10kc/variance'] = model_block_batch_normalize_10kc_variance
        model_block_dense_18so_bias_initializer_zeros_initializer = torch.as_tensor(data=np.asarray(ckpt['model_block/dense_18so/bias']), dtype=torch_types['float32'], device=device)
        model_block_dense_18so_bias = torch.nn.parameter.Parameter(data=model_block_dense_18so_bias_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/dense_18so/bias'] = model_block_dense_18so_bias
        model_block_dense_18so_weights_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['model_block/dense_18so/weights']), dtype=torch_types['float32'], device=device)
        model_block_dense_18so_weights = torch.nn.parameter.Parameter(data=model_block_dense_18so_weights_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/dense_18so/weights'] = model_block_dense_18so_weights
        loss_block_conv_23xc_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        loss_block_conv_23xc_filters = torch.nn.parameter.Parameter(data=loss_block_conv_23xc_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_23xc/filters'] = loss_block_conv_23xc_filters
        loss_block_batch_normalize_27bi_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        loss_block_batch_normalize_27bi_mean = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_27bi_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_27bi/mean'] = loss_block_batch_normalize_27bi_mean
        loss_block_batch_normalize_27bi_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        loss_block_batch_normalize_27bi_offset = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_27bi_offset_initializer_zeros_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_27bi/offset'] = loss_block_batch_normalize_27bi_offset
        loss_block_batch_normalize_27bi_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        loss_block_batch_normalize_27bi_scale = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_27bi_scale_initializer_ones_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_27bi/scale'] = loss_block_batch_normalize_27bi_scale
        loss_block_batch_normalize_27bi_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        loss_block_batch_normalize_27bi_variance = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_27bi_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_27bi/variance'] = loss_block_batch_normalize_27bi_variance
        loss_block_conv_29dy_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 16, 3, 3]))
        loss_block_conv_29dy_filters = torch.nn.parameter.Parameter(data=loss_block_conv_29dy_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_29dy/filters'] = loss_block_conv_29dy_filters
        loss_block_conv_31fo_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 64, 3, 3]))
        loss_block_conv_31fo_filters = torch.nn.parameter.Parameter(data=loss_block_conv_31fo_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_31fo/filters'] = loss_block_conv_31fo_filters
        return trainable_params
    
    @staticmethod
    def model(data_block_input_data, probes, trainable_params, training):
        model_block_conv_6gw = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/conv_6gw/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_relu_8im = torch.nn.functional.relu(input=model_block_conv_6gw, inplace=False)
        model_block_batch_normalize_10kc = torch.nn.functional.batch_norm(input=model_block_relu_8im, running_mean=trainable_params['model_block/batch_normalize_10kc/mean'], running_var=trainable_params['model_block/batch_normalize_10kc/variance'], weight=trainable_params['model_block/batch_normalize_10kc/scale'], bias=trainable_params['model_block/batch_normalize_10kc/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_max_pool2d_12ms = torch.nn.functional.max_pool2d(input=model_block_batch_normalize_10kc, kernel_size=3, stride=1, padding=[0, 0])
        model_block_max_pool2d_14oi = torch.nn.functional.max_pool2d(input=model_block_max_pool2d_12ms, kernel_size=3, stride=1, padding=[0, 0])
        model_block_output = torch.flatten(input=model_block_max_pool2d_14oi, start_dim=1, end_dim=-1)
        model_block_dense_18so = torch.nn.functional.linear(weight=trainable_params['model_block/dense_18so/weights'], bias=trainable_params['model_block/dense_18so/bias'], input=model_block_output)
        model_block_probes = torch.nn.functional.softmax(input=model_block_dense_18so, dim=None)
        probes['model_block/probes'] = model_block_probes
        return model_block_output
    
    @staticmethod
    def get_loss(data_block_input_data, model_block_output, trainable_params, training):
        loss_block_conv_23xc = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['loss_block/conv_23xc/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        loss_block_reluu = torch.nn.functional.relu(input=loss_block_conv_23xc, inplace=False)
        loss_block_batch_normalize_27bi = torch.nn.functional.batch_norm(input=loss_block_reluu, running_mean=trainable_params['loss_block/batch_normalize_27bi/mean'], running_var=trainable_params['loss_block/batch_normalize_27bi/variance'], weight=trainable_params['loss_block/batch_normalize_27bi/scale'], bias=trainable_params['loss_block/batch_normalize_27bi/offset'], training=training, momentum=0.1, eps=0.001)
        loss_block_conv_29dy = torch.nn.functional.conv2d(input=loss_block_batch_normalize_27bi, weight=trainable_params['loss_block/conv_29dy/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_conv_31fo = torch.nn.functional.conv2d(input=loss_block_conv_29dy, weight=trainable_params['loss_block/conv_31fo/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_feature = torch.flatten(input=loss_block_conv_31fo, start_dim=1, end_dim=-1)
        loss_block_cross_0 = torch.nn.functional.mse_loss(input=[loss_block_feature, model_block_output][0], target=[loss_block_feature, model_block_output][1], size_average=None, reduce=None, reduction='mean')
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_6gw/filters', 'model_block/dense_18so/weights', 'loss_block/conv_23xc/filters', 'loss_block/conv_29dy/filters', 'loss_block/conv_31fo/filters'])))
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

C = Checkpoint("examples/configs/small1_linear.yml", 'pytorch', ['checkpoints', 'test_code_gen_ckpt_trained.json'], None)

ckpt = C.load()

model = Model(ckpt)

model.to(device)

trainable_params = model.trainable_params
optimizer = model.get_optimizer(model.params)

learning_rate = model.get_scheduler(optimizer)

probes = dict()

def inference(data_block_input_data, probes, trainable_params):
    
    model.training=False
    training = model.training
    
    preds = model(data_block_input_data, probes, trainable_params, training)
    
    return preds
    
def evaluation(inputs, probes, trainable_params):
    preds = inference(inputs, probes, trainable_params)
    
    
    model.training=False
    training = model.training
    
    loss = model.get_loss(inputs, preds, trainable_params, training)
    return loss
    
    
def train(data_block_input_data, probes, trainable_params):
    
    optimizer.zero_grad()
    model.training=True
    training = model.training
    preds = model(data_block_input_data, probes, trainable_params, training)
    loss = model.get_loss(data_block_input_data, preds, trainable_params, training)
    loss.backward()
    
    
def loop(probes, trainable_params, val_inputs):
    
    for epoch in range(90):
        i = 0
        for data in trainloader:
            inputs, labels = data
    
            inputs = inputs.to(device)
            labels = labels.to(device)
            train(inputs, probes, trainable_params)
            optimizer.step()
    
            if i % 500 == 499:
                results = evaluation(val_inputs, probes, trainable_params)
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

val_inputs, val_labels = iter(valloader).next()

val_inputs = val_inputs.to(device)
val_labels = val_labels.to(device)

loop(probes, trainable_params, val_inputs)

