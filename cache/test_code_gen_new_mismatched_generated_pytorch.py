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
    def get_trainable_params(ckpt, torch_types, device):
        trainable_params = dict()
        model_block_conv_4eg_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[64, 3, 3, 3]))
        model_block_conv_4eg_filters = torch.nn.parameter.Parameter(data=model_block_conv_4eg_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_4eg/filters'] = model_block_conv_4eg_filters
        loss_block_conv_13na_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_13na/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_13na_filters = torch.nn.parameter.Parameter(data=loss_block_conv_13na_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_13na/filters'] = loss_block_conv_13na_filters
        loss_block_batch_normalize_19tw_mean_initializer_zeros_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_19tw/mean']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_19tw_mean = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_19tw_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_19tw/mean'] = loss_block_batch_normalize_19tw_mean
        loss_block_batch_normalize_19tw_offset_initializer_zeros_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_19tw/offset']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_19tw_offset = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_19tw_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['loss_block/batch_normalize_19tw/offset'] = loss_block_batch_normalize_19tw_offset
        loss_block_batch_normalize_19tw_scale_initializer_ones_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_19tw/scale']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_19tw_scale = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_19tw_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['loss_block/batch_normalize_19tw/scale'] = loss_block_batch_normalize_19tw_scale
        loss_block_batch_normalize_19tw_variance_initializer_ones_initializer = torch.as_tensor(data=np.asarray(ckpt['loss_block/batch_normalize_19tw/variance']), dtype=torch_types['float32'], device=device)
        loss_block_batch_normalize_19tw_variance = torch.nn.parameter.Parameter(data=loss_block_batch_normalize_19tw_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['loss_block/batch_normalize_19tw/variance'] = loss_block_batch_normalize_19tw_variance
        loss_block_conv_21vm_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_21vm/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_21vm_filters = torch.nn.parameter.Parameter(data=loss_block_conv_21vm_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_21vm/filters'] = loss_block_conv_21vm_filters
        loss_block_conv_23xc_filters_initializer_xavier_uniform = torch.as_tensor(data=np.asarray(ckpt['loss_block/conv_23xc/filters']), dtype=torch_types['float32'], device=device)
        loss_block_conv_23xc_filters = torch.nn.parameter.Parameter(data=loss_block_conv_23xc_filters_initializer_xavier_uniform, requires_grad=False)
        trainable_params['loss_block/conv_23xc/filters'] = loss_block_conv_23xc_filters
        return trainable_params
    
    @staticmethod
    def model(trainable_params, data_block_input_data):
        model_block_conv_4eg = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/conv_4eg/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        model_block_max_pool2d_6gw = torch.nn.functional.max_pool2d(input=model_block_conv_4eg, kernel_size=3, stride=2, padding=[0, 0])
        model_block_max_pool2d_8im = torch.nn.functional.max_pool2d(input=model_block_max_pool2d_6gw, kernel_size=3, stride=2, padding=[0, 0])
        model_block_output = torch.flatten(input=model_block_max_pool2d_8im, start_dim=1, end_dim=-1)
        return model_block_output 
    
    @staticmethod
    def get_loss(training, trainable_params, inputs, data_block_input_data):
        loss_block_conv_13na = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['loss_block/conv_13na/filters'], bias=None, stride=1, padding=[1, 1], dilation=1, groups=1)
        loss_block_reluu = torch.nn.functional.relu(input=loss_block_conv_13na, inplace=False)
        loss_block_dropout_17rg = torch.nn.functional.dropout(input=loss_block_reluu, p=0.2, training=training, inplace=False)
        loss_block_batch_normalize_19tw = torch.nn.functional.batch_norm(input=loss_block_dropout_17rg, running_mean=trainable_params['loss_block/batch_normalize_19tw/mean'], running_var=trainable_params['loss_block/batch_normalize_19tw/variance'], weight=trainable_params['loss_block/batch_normalize_19tw/scale'], bias=trainable_params['loss_block/batch_normalize_19tw/offset'], training=training, momentum=0.1, eps=0.001)
        loss_block_conv_21vm = torch.nn.functional.conv2d(input=loss_block_batch_normalize_19tw, weight=trainable_params['loss_block/conv_21vm/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_conv_23xc = torch.nn.functional.conv2d(input=loss_block_conv_21vm, weight=trainable_params['loss_block/conv_23xc/filters'], bias=None, stride=1, padding=[0, 0], dilation=1, groups=1)
        loss_block_feature = torch.flatten(input=loss_block_conv_23xc, start_dim=1, end_dim=-1)
        loss_block_cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=inputs[0], input=inputs[1])
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_4eg/filters', 'loss_block/conv_13na/filters', 'loss_block/conv_21vm/filters', 'loss_block/conv_23xc/filters'])))
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

C = Checkpoint("examples/configs/small1_linear.yml", ['checkpoints', 'test_code_gen_ckpt.json'], None)

ckpt = C.load()

model = Model(ckpt)

model.to(device)

trainable_params = model.trainable_params
optimizer = model.get_optimizer(trainable_params)

learning_rate = model.get_scheduler(optimizer)


def inference(trainable_params, data_block_input_data):
    
    preds = torch.max(model(trainable_params, data_block_input_data), 1)
    preds = preds[1]
    return preds
    
def evaluation(trainable_params, training, labels, data_block_input_data):
    
    preds = inference(trainable_params, data_block_input_data)
    
    total = labels.size(0)
    matches = (preds == labels).sum().item()
    perf = matches / total
    
    loss = model.get_loss(training, trainable_params, [labels, preds], data_block_input_data)
    return perf, loss
    
    
def train(trainable_params, training, labels, data_block_input_data):
    
    optimizer.zero_grad()
    preds = model(trainable_params, data_block_input_data)
    loss = model.get_loss(training, trainable_params, [labels, preds], data_block_input_data)
    loss.backward()
    
    
def loop(trainloader, test_inputs, test_labels):
    
    for epoch in range(90):
    
        for i, data in enumerate(trainloader, 0):
    
            inputs, labels = data
    
            inputs = inputs.to(device)
            labels = labels.to(device)
            train(trainable_params, training, labels, data_block_input_data)
            optimizer.step()
    
            if i % 500 == 499:
                results = evaluation(trainable_params, training, labels, data_block_input_data)
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


