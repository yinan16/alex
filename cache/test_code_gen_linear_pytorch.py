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

    def forward(self, data_block_input_data, trainable_params):
        x = self.model(data_block_input_data, trainable_params)
        return x

    @staticmethod
    def get_trainable_params(ckpt):
        trainable_params = dict()
        model_block_conv_4eg_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 3, 3, 3]))
        model_block_conv_4eg_filters = torch.nn.parameter.Parameter(data=model_block_conv_4eg_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_4eg/filters'] = model_block_conv_4eg_filters
        loss_block_conv_13na_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_13na/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_13na_filters = torch.nn.parameter.Parameter(data=loss_block_conv_13na_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_13na/filters'] = loss_block_conv_13na_filters
        loss_block_batch_normalize_17rg_mean_initializer_zeros_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_17rg/mean']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_17rg_mean = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_17rg_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_17rg/mean'] = loss_block_batch_normalize_17rg_mean
        loss_block_batch_normalize_17rg_offset_initializer_zeros_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_17rg/offset']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_17rg_offset = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_17rg_offset_initializer_zeros_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_17rg/offset'] = loss_block_batch_normalize_17rg_offset
        loss_block_batch_normalize_17rg_scale_initializer_ones_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_17rg/scale']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_17rg_scale = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_17rg_scale_initializer_ones_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_17rg/scale'] = loss_block_batch_normalize_17rg_scale
        loss_block_batch_normalize_17rg_variance_initializer_ones_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_17rg/variance']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_17rg_variance = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_17rg_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_17rg/variance'] = loss_block_batch_normalize_17rg_variance
        loss_block_conv_19tw_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_19tw/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_19tw_filters = torch.nn.parameter.Parameter(data=loss_block_conv_19tw_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_19tw/filters'] = loss_block_conv_19tw_filters
        loss_block_conv_21vm_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_21vm/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_21vm_filters = torch.nn.parameter.Parameter(data=loss_block_conv_21vm_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_21vm/filters'] = loss_block_conv_21vm_filters
        return trainable_params

    @staticmethod
    def model(data_block_input_data, trainable_params):
        model_block_conv_4eg = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/conv_4eg/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_max_pool2d_6gw = torch.nn.functional.max_pool2d(input=model_block_conv_4eg, kernel_size=3, stride=1, padding=[0, 0])
        model_block_max_pool2d_8im = torch.nn.functional.max_pool2d(input=model_block_max_pool2d_6gw, kernel_size=3, stride=1, padding=[0, 0])
        model_block_output = torch.flatten(input=model_block_max_pool2d_8im, start_dim=1, end_dim=-1)
        return model_block_output

    @staticmethod
    def get_loss(data_block_input_data, training, model_block_output, trainable_params):
        loss_block_conv_13na = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['loss_block/conv_13na/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        loss_block_reluu = torch.nn.functional.relu(input=loss_block_conv_13na, inplace=False)
        loss_block_batch_normalize_17rg = torch.nn.functional.batch_norm(input=loss_block_reluu, running_mean=trainable_params['loss_block/batch_normalize_17rg/mean'], running_var=trainable_params['loss_block/batch_normalize_17rg/variance'], weight=trainable_params['loss_block/batch_normalize_17rg/scale'], bias=trainable_params['loss_block/batch_normalize_17rg/offset'], training=training, momentum=0.1, eps=0.001)
        loss_block_conv_19tw = torch.nn.functional.conv2d(input=loss_block_batch_normalize_17rg, weight=trainable_params['loss_block/conv_19tw/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_conv_21vm = torch.nn.functional.conv2d(input=loss_block_conv_19tw, weight=trainable_params['loss_block/conv_21vm/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_feature = torch.flatten(input=loss_block_conv_21vm, start_dim=1, end_dim=-1)
        loss_block_cross_0 = torch.nn.functional.mse_loss(input=[loss_block_feature, model_block_output][0], target=[loss_block_feature, model_block_output][1], size_average=None, reduce=None, reduction='mean')
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_4eg/filters', 'loss_block/conv_13na/filters', 'loss_block/conv_19tw/filters', 'loss_block/conv_21vm/filters'])))
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

C = Checkpoint("examples/configs/small1_linear.yml", ['checkpoints', 'test_code_gen_ckpt_trained.json'], None)

ckpt = C.load()

model = Model(ckpt)

model.to(device)

trainable_params = model.trainable_params
optimizer = model.get_optimizer(model.params)

learning_rate = model.get_scheduler(optimizer)

probes = dict()

def inference(data_block_input_data, trainable_params):

    model.training=False
    training = model.training

    preds = model(data_block_input_data, trainable_params)

    return preds

def evaluation(data_block_input_data, trainable_params):

    preds = inference(data_block_input_data, trainable_params)

    model.training=False
    training = model.training

    loss = model.get_loss(data_block_input_data, training, preds, trainable_params)
    return loss


def train(data_block_input_data, trainable_params):

    optimizer.zero_grad()
    model.training=True
    training = model.training
    preds = model(data_block_input_data, trainable_params)
    loss = model.get_loss(data_block_input_data, training, preds, trainable_params)
    loss.backward()


def loop(val_inputs, trainable_params):

    for epoch in range(90):
        i = 0
        for data in trainloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            train(inputs, trainable_params)
            optimizer.step()

            if i % 500 == 499:
                results = evaluation(val_inputs, trainable_params)
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

loop(val_inputs, trainable_params)
