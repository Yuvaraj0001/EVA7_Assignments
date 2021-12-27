# The Dawn of Transformers

## Assignment 12:

To implement a `Spatial transformer` on `CIFAR-10` dataset for `50 epochs` and a explanation on what Spatial Transformer does.

## Spatial transformer networks

`Spatial transformer networks` are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. 

`For example`, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations.

#### Spatial transformer networks boils down to three main components:

1. `The localization network` is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
2. `The grid generator` generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. `The sampler` uses the parameters of the transformation and applies it to the input image.

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_12/Spacial_Transformers/Images/stn-arch.png)

Colab Notebook https://colab.research.google.com/drive/1RFP-cf-9Gk8kuBTqmSvP-DT3AfIWtu-K#scrollTo=9lOKxxZMOw_y

GitHub Notebook https://colab.research.google.com/drive/1RFP-cf-9Gk8kuBTqmSvP-DT3AfIWtu-K#scrollTo=9lOKxxZMOw_y

### Model Architecture
Net(
  (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=800, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=2048, out_features=256, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=256, out_features=6, bias=True)
  )
)
### Training logs
Train Epoch: 1 [0/50000 (0%)]	Loss: 2.304312
Train Epoch: 1 [32000/50000 (64%)]	Loss: 2.036470
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 1.8522, Accuracy: 3461/10000 (35%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 1.921410
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.809385

Test set: Average loss: 1.6283, Accuracy: 4141/10000 (41%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.735710
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.618648

Test set: Average loss: 1.5274, Accuracy: 4568/10000 (46%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.852861
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.545522

Test set: Average loss: 1.5124, Accuracy: 4519/10000 (45%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.571889
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.438687

Test set: Average loss: 1.5161, Accuracy: 4651/10000 (47%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.692336
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.477948

Test set: Average loss: 1.3296, Accuracy: 5238/10000 (52%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.782354
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.714799

Test set: Average loss: 1.3055, Accuracy: 5468/10000 (55%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.188083
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.556775

Test set: Average loss: 1.3282, Accuracy: 5280/10000 (53%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.619820
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.286268

Test set: Average loss: 1.2230, Accuracy: 5683/10000 (57%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.170184
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.378503

Test set: Average loss: 1.2146, Accuracy: 5740/10000 (57%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.212034
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.284691

Test set: Average loss: 1.2541, Accuracy: 5595/10000 (56%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.592365
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.281507

Test set: Average loss: 1.2723, Accuracy: 5582/10000 (56%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.391559
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.324125

Test set: Average loss: 1.1350, Accuracy: 6068/10000 (61%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.373469
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.331969

Test set: Average loss: 1.1826, Accuracy: 5899/10000 (59%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.172836
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.219021

Test set: Average loss: 1.2640, Accuracy: 5625/10000 (56%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.472123
Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.466178

Test set: Average loss: 1.1010, Accuracy: 6188/10000 (62%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.240313
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.094283

Test set: Average loss: 1.1003, Accuracy: 6210/10000 (62%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.117773
Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.172345

Test set: Average loss: 1.0977, Accuracy: 6203/10000 (62%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 1.250970
Train Epoch: 19 [32000/50000 (64%)]	Loss: 0.941945

Test set: Average loss: 1.1039, Accuracy: 6108/10000 (61%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 0.963680
Train Epoch: 20 [32000/50000 (64%)]	Loss: 0.926079

Test set: Average loss: 1.0720, Accuracy: 6316/10000 (63%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 1.268942
Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.046438

Test set: Average loss: 1.0408, Accuracy: 6457/10000 (65%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 1.066251
Train Epoch: 22 [32000/50000 (64%)]	Loss: 1.049376

Test set: Average loss: 1.0415, Accuracy: 6415/10000 (64%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 1.171061
Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.020193

Test set: Average loss: 1.0397, Accuracy: 6432/10000 (64%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 0.800569
Train Epoch: 24 [32000/50000 (64%)]	Loss: 1.352759

Test set: Average loss: 1.1139, Accuracy: 6090/10000 (61%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 0.867476
Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.029550

Test set: Average loss: 1.0640, Accuracy: 6342/10000 (63%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 0.837024
Train Epoch: 26 [32000/50000 (64%)]	Loss: 0.909539

Test set: Average loss: 1.0183, Accuracy: 6570/10000 (66%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 0.988625
Train Epoch: 27 [32000/50000 (64%)]	Loss: 1.136986

Test set: Average loss: 1.0365, Accuracy: 6423/10000 (64%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 0.805244
Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.917130

Test set: Average loss: 1.0296, Accuracy: 6457/10000 (65%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 0.962395
Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.903990

Test set: Average loss: 1.0663, Accuracy: 6368/10000 (64%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 0.822480
Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.676228

Test set: Average loss: 1.1673, Accuracy: 6008/10000 (60%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 1.029220
Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.887320

Test set: Average loss: 1.0041, Accuracy: 6537/10000 (65%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 0.859324
Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.979822

Test set: Average loss: 1.0365, Accuracy: 6437/10000 (64%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 0.812233
Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.773517

Test set: Average loss: 1.0234, Accuracy: 6501/10000 (65%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 0.922813
Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.819481

Test set: Average loss: 1.0009, Accuracy: 6582/10000 (66%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 0.887563
Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.788327

Test set: Average loss: 1.0067, Accuracy: 6545/10000 (65%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 1.070144
Train Epoch: 36 [32000/50000 (64%)]	Loss: 1.019018

Test set: Average loss: 1.0952, Accuracy: 6295/10000 (63%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 0.934523
Train Epoch: 37 [32000/50000 (64%)]	Loss: 1.064816

Test set: Average loss: 1.0022, Accuracy: 6580/10000 (66%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 0.686067
Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.203434

Test set: Average loss: 0.9862, Accuracy: 6607/10000 (66%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 0.750939
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.703440

Test set: Average loss: 1.0449, Accuracy: 6436/10000 (64%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 0.839775
Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.845010

Test set: Average loss: 1.0739, Accuracy: 6340/10000 (63%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 0.781632
Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.829711

Test set: Average loss: 1.0635, Accuracy: 6398/10000 (64%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 0.743997
Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.781154

Test set: Average loss: 1.0019, Accuracy: 6591/10000 (66%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 0.828098
Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.793791

Test set: Average loss: 1.0876, Accuracy: 6319/10000 (63%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 0.715948
Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.837755

Test set: Average loss: 0.9941, Accuracy: 6636/10000 (66%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 0.567005
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.709806

Test set: Average loss: 1.0223, Accuracy: 6583/10000 (66%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 0.589294
Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.702880

Test set: Average loss: 1.0321, Accuracy: 6532/10000 (65%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 0.654814
Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.849711

Test set: Average loss: 1.0189, Accuracy: 6560/10000 (66%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 0.930389
Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.533365

Test set: Average loss: 1.2287, Accuracy: 5904/10000 (59%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 1.077944
Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.729898

Test set: Average loss: 1.0270, Accuracy: 6550/10000 (66%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 0.753977
Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.663541

Test set: Average loss: 1.0142, Accuracy: 6589/10000 (66%)
### Visualize STN Results

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_12/Spacial_Transformers/Images/STN_result.png)

### Reference

https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html
