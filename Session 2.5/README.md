# Session 2.5 : PyTorch 101

### Write a neural network that can:

1. take 2 inputs:
    1. an image from the MNIST dataset (say 5), and
    2. a random number between 0 and 9, (say 7)
2. and gives two outputs:
    1. the "number" that was represented by the MNIST image (predict 5), and
    2. the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
    
![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%202.5/Images/network.png)    
    
3. you can mix fully connected layers and convolution layers
4. you can use one-hot encoding to represent the random number input as well as the "summed" output.
    1. Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
    2. Sum (13) can be represented as:
        1. 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
        2. 0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010

### Dataset preparation 

The `MNIST database` of handwritten digits has a training set of 60,000 images and training set of 10,000 images.
Each images are grayscale and size of 28x28 pixels.

Another input is `random number` generated between 0 and 9.

### Model

We define our own class `class Net(nn.Module)` and we inharite nn.Module which is Base class for all neural network modules. Then we define initialize function `__init__` after we inherite all the functionality of nn.Module in our class `super(Net, self).__init__()`. After that we start building our model.

`nn.Conv2d`: Applies a 2D convolution over an input image.

`nn.MaxPool2d`: Applies a 2D max pooling over an input image.

The `forward()` function defines the process of calculating the output using the given layers and activation functions.

`F.relu` :  the rectified linear unit function 

`x.view(-1, 10)`:The view method returns a tensor with the same data as the self tensor but with a different shape. First parameter represent the `batch_size` in our case batch_size is 128, if you don't know the batch_size pass -1 tensor.view method will take care of batch_size for you. Second parameter is the `column or the number of neurons` you want.

`x1 = torch.cat((x, randomNumber), dim=1)` : concatenate second input (random number) to the output from flattening.
One hot encoded vector of 10 elements, is passed through a fully connected layer with 20 output features

`F.log_softmax(x1,dim=1)` and `F.log_softmax(x1,dim=1)`: log_softmax is an activation function. Here, two outputs are returning. 


```python
a=Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
  (conv6): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
  (conv7): Conv2d(1024, 10, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=20, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=19, bias=True)
)
```

### Training Log

`Loss` is calculated by sum of bot MNIST and sum loss.The MNIST and Sum `accuracy` are below,


```python
Epoch 1 : 
Train set: loss: 2.0403
Val set: loss: 1.040, MNist Accuracy:98.26, Sum_Accuracy:35.22

Epoch 2 : 
Train set: loss: 1.2970
Val set: loss: 0.572, MNist Accuracy:98.76, Sum_Accuracy:89.2

Epoch 3 : 
Train set: loss: 0.3346
Val set: loss: 0.187, MNist Accuracy:98.92, Sum_Accuracy:98.52

Epoch 4 : 
Train set: loss: 0.1684
Val set: loss: 0.105, MNist Accuracy:98.88, Sum_Accuracy:98.62

Epoch 5 : 
Train set: loss: 0.0975
Val set: loss: 0.065, MNist Accuracy:99.14, Sum_Accuracy:99.0
```

### Demo result


```python
image,_,_,_ = test_dataset[random.randint(1,10000)]
rnum=random.randint(0,9)
mnist_pred, sum_pred =demo_prediction(image,rnum)
```

Mnist Prediction: 9                                                                                                             
Random Number Generated: 6                                                                                                     
Sum: 15                                                                                                                              
![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%202.5/Images/image.png)



```python

```
