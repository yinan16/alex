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
