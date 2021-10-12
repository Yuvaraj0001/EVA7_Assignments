# Session 1: Background & Basics: Neural Architecture

### Import Necessary Libraries


```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
```

`from __future__ import print_function` : To add this at the top of each module for disabling the statement and use the print() function, use this future statement at the top of your module.

`import torch` : The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors.

`import torch.nn as nn` : This package provide us many more classes and modules to implement and train the neural network.

`import torch.nn.functional as F` : This package contains all the functions in the `torch.nn` library as well as a wide range of loss and activation functions

`import torch.optim as optim` : This package implementing various optimization algorithms. Contains optimizers such as Stochastic Gradient Descent (SGD), which update the weights of Parameter during the backward step.

`from torchvision import datasets, transforms` : The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.

### Building the Network


```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)    #input-28x28  Output-28x28   RF-3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   #input-28x28  Output-28x28   RF-5x5
        self.pool1 = nn.MaxPool2d(2, 2)                #input-28x28  Output-14x14   RF-10x10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  #input-14x14  Output-14x14   RF-12x12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #input-14x14  Output-14x14   RF-14x14
        self.pool2 = nn.MaxPool2d(2, 2)                #input-14x14  Output-7x7     RF-28x28
        self.conv5 = nn.Conv2d(256, 512, 3)            #input-7x7    Output-5x5     RF-30x30
        self.conv6 = nn.Conv2d(512, 1024, 3)           #input-5x5    Output-3x3     RF-32x32
        self.conv7 = nn.Conv2d(1024, 10, 3)            #input-3x3    Output-1x1     RF-34x34

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
```

We define our own class `class Net(nn.Module)` and we inharite nn.Module which is Base class for all neural network modules. Then we define initialize function `__init__` after we inherite all the functionality of nn.Module in our class `super(Net, self).__init__()`. After that we start building our model.

`nn.Conv2d`: Applies a 2D convolution over an input image.

`nn.MaxPool2d`: Applies a 2D max pooling over an input image.

The `forward()` function defines the process of calculating the output using the given layers and activation functions.

`F.relu` :  the rectified linear unit function 

`x.view(-1, 10)`:The view method returns a tensor with the same data as the self tensor but with a different shape. First parameter represent the `batch_size` in our case batch_size is 128, if you don't know the batch_size pass -1 tensor.view method will take care of batch_size for you. Second parameter is the `column or the number of neurons` you want.

`F.log_softmax(x)`: log_softmax is an activation function.





```python
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

`from torchsummary import summary` Torch-summary provides information complementary to what is provided by `print(your_model)` in PyTorch.

`torch.cuda`: This package adds support for CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation. Use `is_available()` to determine if your system supports CUDA.

`torch.device`: Context-manager that changes the selected device.

`model = Net().to(device)`: load model to available device

`summary(model, input_size=(1, 28, 28))`: Summarize the given PyTorch model. Summarized information includes Layer names, input/output shapes, kernel shape, No. of parameters, etc.

### Preparing the Dataset


```python
torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
```

`torch.manual_seed(1)`:Sets the seed for generating random numbers. Returns a `torch.Generator` object.

`batch_size`: batch size is the number of images (here, 128 images) we want to read in one go.

`kwargs = {'num_workers': 1, 'pin_memory': True}`: `num_workers`-how many subprocesses to use for data loading. For data loading, passing `pin_memory=True` to a DataLoader will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.

`torch.utils.data.DataLoader`: we make Data iterable by loading it to a loader.

`datasets.MNIST`: Downloading the MNIST dataset for training and testing at path `../data`.

`transform=transforms.Compose`:Composes several transforms together. This transform does not support torchscript.

`transforms.ToTensor()`: This converts the image into numbers, that are understandable by the system. It separates the image into three color channels (separate images): red, green & blue. Then it converts the pixels of each image to the brightness of their color between 0 and 255. These values are then scaled down to a range between 0 and 1. The image is now a Torch Tensor.

`transforms.Normalize((0.1307,), (0.3081,))`: This normalizes the tensor with a mean `(0.1307,)` and standard deviation `(0.3081,)` which goes as the two parameters respectively.

`shuffle=True`: Shuffle the training data to make it independent of the order by making it a `True`.


### Training and Testing


```python
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

`tqdm`:It means “progress”. Instantly make your loops show a smart progress meter - just wrap any iterable with `tqdm(iterable)`. `tqdm` uses smart algorithms to predict the remaining time and to skip unnecessary iteration displays, which allows for a negligible overhead in most cases.

In the above cell, train and test functions are generated.

`zero_grad`: Sets the gradients of all optimized `torch.Tensor` to zero.

`F.nll_loss`:The negative log likelihood loss. It is useful to train a classification problem with C classes.

`loss.backward()`:Computes the derivative of the loss with respect to the parameters

`optimizer.step()`:All optimizers implement a step() method, that updates the parameters.

### Training Model


```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```
0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:20: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=1.99245023727417 batch_id=468: 100%|██████████| 469/469 [00:37<00:00, 12.38it/s]

Test set: Average loss: 1.9687, Accuracy: 2790/10000 (28%)
This is main code. The neural network iterates over the training set and updates the weights. We make use of `optim.SGD` which is a Stochastic Gradient Descent (SGD) provided by PyTorch to optimize the model, perform gradient descent and update the weights by back-propagation. Thus in each `epoch` (number of times we iterate over the training set), we will be seeing a gradual decrease in training loss.


```python

```
