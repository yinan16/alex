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
        xavier_uniform_a0a939d4_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['conv_5fo/filters']), dtype=torch_types['float32'], device=device)
        conv_5fo_filters = torch.nn.parameter.Parameter(data=xavier_uniform_a0a939d4_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['conv_5fo/filters'] = conv_5fo_filters
        zeros_initializer_a0a939f0_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['batch_normalize_11lk/mean']), dtype=torch_types['float32'], device=device)
        batch_normalize_11lk_mean = torch.nn.parameter.Parameter(data=zeros_initializer_a0a939f0_d1aa_11eb_99c7_8facf688c982, requires_grad=False)
        trainable_params['batch_normalize_11lk/mean'] = batch_normalize_11lk_mean
        zeros_initializer_a0a939fa_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['batch_normalize_11lk/offset']), dtype=torch_types['float32'], device=device)
        batch_normalize_11lk_offset = torch.nn.parameter.Parameter(data=zeros_initializer_a0a939fa_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['batch_normalize_11lk/offset'] = batch_normalize_11lk_offset
        ones_initializer_a0a93a02_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['batch_normalize_11lk/scale']), dtype=torch_types['float32'], device=device)
        batch_normalize_11lk_scale = torch.nn.parameter.Parameter(data=ones_initializer_a0a93a02_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['batch_normalize_11lk/scale'] = batch_normalize_11lk_scale
        ones_initializer_a0a93a0a_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['batch_normalize_11lk/variance']), dtype=torch_types['float32'], device=device)
        batch_normalize_11lk_variance = torch.nn.parameter.Parameter(data=ones_initializer_a0a93a0a_d1aa_11eb_99c7_8facf688c982, requires_grad=False)
        trainable_params['batch_normalize_11lk/variance'] = batch_normalize_11lk_variance
        xavier_uniform_a0a93a14_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['conv_13na/filters']), dtype=torch_types['float32'], device=device)
        conv_13na_filters = torch.nn.parameter.Parameter(data=xavier_uniform_a0a93a14_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['conv_13na/filters'] = conv_13na_filters
        xavier_uniform_a0a93a2e_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['conv_15pq/filters']), dtype=torch_types['float32'], device=device)
        conv_15pq_filters = torch.nn.parameter.Parameter(data=xavier_uniform_a0a93a2e_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['conv_15pq/filters'] = conv_15pq_filters
        zeros_initializer_a0a93a46_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['dense_19tw/bias']), dtype=torch_types['float32'], device=device)
        dense_19tw_bias = torch.nn.parameter.Parameter(data=zeros_initializer_a0a93a46_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
        trainable_params['dense_19tw/bias'] = dense_19tw_bias
        xavier_uniform_a0a93a4e_d1aa_11eb_99c7_8facf688c982 = torch.as_tensor(data=np.asarray(ckpt['dense_19tw/weights']), dtype=torch_types['float32'], device=device)
        dense_19tw_weights = torch.nn.parameter.Parameter(data=xavier_uniform_a0a93a4e_d1aa_11eb_99c7_8facf688c982, requires_grad=True)
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
        exponential_decay_a0a93a60_d1aa_11eb_99c7_8facf688c982 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1, verbose=False)
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

