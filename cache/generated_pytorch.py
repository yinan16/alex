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
    def get_trainable_params():
        trainable_params = dict()
        model_block_conv_6gw_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 3, 3, 3]))
        model_block_conv_6gw_filters = torch.nn.parameter.Parameter(data=model_block_conv_6gw_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
        model_block_batch_normalize_12ms_mean_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_12ms_mean = torch.nn.parameter.Parameter(data=model_block_batch_normalize_12ms_mean_initializer_zeros_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_12ms/mean'] = model_block_batch_normalize_12ms_mean
        model_block_batch_normalize_12ms_offset_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_12ms_offset = torch.nn.parameter.Parameter(data=model_block_batch_normalize_12ms_offset_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_12ms/offset'] = model_block_batch_normalize_12ms_offset
        model_block_batch_normalize_12ms_scale_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_12ms_scale = torch.nn.parameter.Parameter(data=model_block_batch_normalize_12ms_scale_initializer_ones_initializer, requires_grad=True)
        trainable_params['model_block/batch_normalize_12ms/scale'] = model_block_batch_normalize_12ms_scale
        model_block_batch_normalize_12ms_variance_initializer_ones_initializer = torch.nn.init.ones_(tensor=torch.empty(*[16, ]))
        model_block_batch_normalize_12ms_variance = torch.nn.parameter.Parameter(data=model_block_batch_normalize_12ms_variance_initializer_ones_initializer, requires_grad=False)
        trainable_params['model_block/batch_normalize_12ms/variance'] = model_block_batch_normalize_12ms_variance
        model_block_conv_14oi_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 3, 3]))
        model_block_conv_14oi_filters = torch.nn.parameter.Parameter(data=model_block_conv_14oi_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_14oi/filters'] = model_block_conv_14oi_filters
        model_block_conv_16qy_filters_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[16, 16, 5, 5]))
        model_block_conv_16qy_filters = torch.nn.parameter.Parameter(data=model_block_conv_16qy_filters_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/conv_16qy/filters'] = model_block_conv_16qy_filters
        model_block_dense_20ue_bias_initializer_zeros_initializer = torch.nn.init.zeros_(tensor=torch.empty(*[1, ]))
        model_block_dense_20ue_bias = torch.nn.parameter.Parameter(data=model_block_dense_20ue_bias_initializer_zeros_initializer, requires_grad=True)
        trainable_params['model_block/dense_20ue/bias'] = model_block_dense_20ue_bias
        model_block_dense_20ue_weights_initializer_xavier_uniform = torch.nn.init.xavier_uniform_(tensor=torch.empty(*[10, 16384]))
        model_block_dense_20ue_weights = torch.nn.parameter.Parameter(data=model_block_dense_20ue_weights_initializer_xavier_uniform, requires_grad=True)
        trainable_params['model_block/dense_20ue/weights'] = model_block_dense_20ue_weights
        return trainable_params
    
    @staticmethod
    def model(data_block_input_data, trainable_params, training):
        model_block_conv_6gw = torch.nn.functional.conv2d(input=data_block_input_data, weight=trainable_params['model_block/conv_6gw/filters'], bias=None, stride=1, padding='same', dilation=1, groups=1)
        model_block_reluu = torch.nn.functional.relu(input=model_block_conv_6gw, inplace=False)
        model_block_dropout_10kc = torch.nn.functional.dropout(input=model_block_reluu, p=0.2, training=training, inplace=False)
        model_block_batch_normalize_12ms = torch.nn.functional.batch_norm(input=model_block_dropout_10kc, running_mean=trainable_params['model_block/batch_normalize_12ms/mean'], running_var=trainable_params['model_block/batch_normalize_12ms/variance'], weight=trainable_params['model_block/batch_normalize_12ms/scale'], bias=trainable_params['model_block/batch_normalize_12ms/offset'], training=training, momentum=0.1, eps=0.001)
        model_block_conv_14oi = torch.nn.functional.conv2d(input=model_block_batch_normalize_12ms, weight=trainable_params['model_block/conv_14oi/filters'], bias=None, stride=1, padding='same', dilation=1, groups=1)
        model_block_conv_16qy = torch.nn.functional.conv2d(input=model_block_conv_14oi, weight=trainable_params['model_block/conv_16qy/filters'], bias=None, stride=1, padding='same', dilation=1, groups=1)
        model_block_flatten_18so = torch.flatten(input=model_block_conv_16qy, start_dim=1, end_dim=-1)
        model_block_dense_20ue = torch.nn.functional.linear(weight=trainable_params['model_block/dense_20ue/weights'], bias=trainable_params['model_block/dense_20ue/bias'], input=model_block_flatten_18so)
        model_block_d_1 = torch.nn.functional.softmax(input=model_block_dense_20ue, dim=None)
        return model_block_d_1 
    
    @staticmethod
    def get_loss(trainable_params, inputs):
        loss_block_cross_0 = torch.nn.functional.cross_entropy(weight=None, ignore_index=-100, reduction='mean', target=inputs[0], input=inputs[1])
        loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_6gw/filters', 'model_block/conv_14oi/filters', 'model_block/conv_16qy/filters', 'model_block/dense_20ue/weights'])))
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
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

test_images, test_labels = iter(testloader).next()

def test(images, labels):
    correct = 0
    total = 0
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images, False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #     100 * correct / total))
    return correct / total


# from alex.alex.checkpoint import Checkpoint

# C = Checkpoint("examples/configs/small1.yml",
#                ["cache",  "config_1622420349826577.json"],
#                ["checkpoints", None])

# ckpt = C.load()

model = Model()

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
        if i % 500 == 499:    # print every 2000 mini-batches
            accuracy = test(test_images, test_labels)
            print('[%d, %5d] accuracy: %.3f, loss: %.3f' %
                  (epoch + 1, i + 1, accuracy, running_loss / 500))
            # C.save(model.trainable_params)
            running_loss = 0.0
    learning_rate.step()
print('Finished Training')


