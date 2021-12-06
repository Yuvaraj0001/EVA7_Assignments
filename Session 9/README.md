# RESNETS AND HIGHER RECEPTIVE FIELDS

Built custom ResNet Model on CIFAR data and achieved the test accuracy of 90.88% in 24 Epochs.

## Group:

1. Yuvaraj V (yuvaraj100493@gmail.com)
2. Satwik Mishra (satwiksmishra@gmail.com)

## Custom ResNet Model

No of Parameters - 99.85k

Highest Train Accuracy -70.49

Highest test Accuracy - 77.75

----------------------------------------------------------------

        Layer (type)               Output Shape         Param #
        
================================================================

            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
         Dropout2d-3           [-1, 64, 32, 32]               0
       ConvBNBlock-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
         Dropout2d-8          [-1, 128, 16, 16]               0
   TransitionBlock-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
        Dropout2d-12          [-1, 128, 16, 16]               0
      ConvBNBlock-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
        Dropout2d-16          [-1, 128, 16, 16]               0
      ConvBNBlock-17          [-1, 128, 16, 16]               0
         ResBlock-18          [-1, 128, 16, 16]               0
           Conv2d-19          [-1, 256, 16, 16]         294,912
        MaxPool2d-20            [-1, 256, 8, 8]               0
      BatchNorm2d-21            [-1, 256, 8, 8]             512
        Dropout2d-22            [-1, 256, 8, 8]               0
  TransitionBlock-23            [-1, 256, 8, 8]               0
           Conv2d-24            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-25            [-1, 512, 4, 4]               0
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
        Dropout2d-27            [-1, 512, 4, 4]               0
  TransitionBlock-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-30            [-1, 512, 4, 4]           1,024
        Dropout2d-31            [-1, 512, 4, 4]               0
      ConvBNBlock-32            [-1, 512, 4, 4]               0
           Conv2d-33            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-34            [-1, 512, 4, 4]           1,024
        Dropout2d-35            [-1, 512, 4, 4]               0
      ConvBNBlock-36            [-1, 512, 4, 4]               0
         ResBlock-37            [-1, 512, 4, 4]               0
        MaxPool2d-38            [-1, 512, 1, 1]               0
           Linear-39                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.32
Params size (MB): 25.07
Estimated Total Size (MB): 33.40
----------------------------------------------------------------

## LR Search Plot

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/LR%20plot.png)

## Training and Testing Logs
EPOCH: 1
  0%|          | 0/98 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
/content/drive/My Drive/CRN/customresnet.py:85: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(out)
Loss=1.2821202278137207 Batch_id=97 Accuracy=40.43: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set: Average loss: 1.3086, Accuracy: 5399/10000 (53.99%)

Test Accuracy: 53.99 has increased. Saving the model

EPOCH: 2
Loss=1.3674408197402954 Batch_id=97 Accuracy=57.37: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 1.2467, Accuracy: 6043/10000 (60.43%)

Test Accuracy: 60.43 has increased. Saving the model

EPOCH: 3
Loss=1.0467630624771118 Batch_id=97 Accuracy=64.95: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 1.0122, Accuracy: 6807/10000 (68.07%)

Test Accuracy: 68.07 has increased. Saving the model

EPOCH: 4
Loss=0.8910414576530457 Batch_id=97 Accuracy=67.84: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 0.8603, Accuracy: 7287/10000 (72.87%)

Test Accuracy: 72.87 has increased. Saving the model

EPOCH: 5
Loss=0.9103121757507324 Batch_id=97 Accuracy=74.65: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 1.1098, Accuracy: 6995/10000 (69.95%)

EPOCH: 6
Loss=0.564529538154602 Batch_id=97 Accuracy=77.42: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 0.7642, Accuracy: 7706/10000 (77.06%)

Test Accuracy: 77.06 has increased. Saving the model

EPOCH: 7
Loss=0.5887299180030823 Batch_id=97 Accuracy=81.33: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set: Average loss: 0.6437, Accuracy: 8001/10000 (80.01%)

Test Accuracy: 80.01 has increased. Saving the model

EPOCH: 8
Loss=0.33342957496643066 Batch_id=97 Accuracy=83.60: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set: Average loss: 0.6496, Accuracy: 7972/10000 (79.72%)

EPOCH: 9
Loss=0.4106210470199585 Batch_id=97 Accuracy=85.87: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set: Average loss: 0.4620, Accuracy: 8514/10000 (85.14%)

Test Accuracy: 85.14 has increased. Saving the model

EPOCH: 10
Loss=0.3624134957790375 Batch_id=97 Accuracy=87.35: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.4397, Accuracy: 8571/10000 (85.71%)

Test Accuracy: 85.71 has increased. Saving the model

EPOCH: 11
Loss=0.323405385017395 Batch_id=97 Accuracy=88.90: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.4165, Accuracy: 8650/10000 (86.50%)

Test Accuracy: 86.5 has increased. Saving the model

EPOCH: 12
Loss=0.34409767389297485 Batch_id=97 Accuracy=90.30: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.4247, Accuracy: 8697/10000 (86.97%)

Test Accuracy: 86.97 has increased. Saving the model

EPOCH: 13
Loss=0.2811940610408783 Batch_id=97 Accuracy=91.14: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.4155, Accuracy: 8788/10000 (87.88%)

Test Accuracy: 87.88 has increased. Saving the model

EPOCH: 14
Loss=0.22680529952049255 Batch_id=97 Accuracy=92.17: 100%|██████████| 98/98 [00:56<00:00,  1.75it/s]

Test set: Average loss: 0.3654, Accuracy: 8876/10000 (88.76%)

Test Accuracy: 88.76 has increased. Saving the model

EPOCH: 15
Loss=0.18747879564762115 Batch_id=97 Accuracy=93.15: 100%|██████████| 98/98 [00:56<00:00,  1.75it/s]

Test set: Average loss: 0.3601, Accuracy: 8879/10000 (88.79%)

Test Accuracy: 88.79 has increased. Saving the model

EPOCH: 16
Loss=0.1353045552968979 Batch_id=97 Accuracy=93.98: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3648, Accuracy: 8909/10000 (89.09%)

Test Accuracy: 89.09 has increased. Saving the model

EPOCH: 17
Loss=0.15059086680412292 Batch_id=97 Accuracy=94.42: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3353, Accuracy: 8994/10000 (89.94%)

Test Accuracy: 89.94 has increased. Saving the model

EPOCH: 18
Loss=0.11599750816822052 Batch_id=97 Accuracy=95.06: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3181, Accuracy: 9005/10000 (90.05%)

Test Accuracy: 90.05 has increased. Saving the model

EPOCH: 19
Loss=0.13956400752067566 Batch_id=97 Accuracy=95.56: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3095, Accuracy: 9054/10000 (90.54%)

Test Accuracy: 90.54 has increased. Saving the model

EPOCH: 20
Loss=0.15439008176326752 Batch_id=97 Accuracy=95.78: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3193, Accuracy: 9035/10000 (90.35%)

EPOCH: 21
Loss=0.13163802027702332 Batch_id=97 Accuracy=95.74: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3093, Accuracy: 9056/10000 (90.56%)

Test Accuracy: 90.56 has increased. Saving the model

EPOCH: 22
Loss=0.15073522925376892 Batch_id=97 Accuracy=95.96: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3101, Accuracy: 9052/10000 (90.52%)

EPOCH: 23
Loss=0.06864310055971146 Batch_id=97 Accuracy=96.21: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3107, Accuracy: 9066/10000 (90.66%)

Test Accuracy: 90.66 has increased. Saving the model

EPOCH: 24
Loss=0.14406953752040863 Batch_id=97 Accuracy=96.19: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set: Average loss: 0.3042, Accuracy: 9088/10000 (90.88%)

Test Accuracy: 90.88 has increased. Saving the model

## Accuracy Plot

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/Accuracy.png)

Accuracy of plane : 85 %

Accuracy of   car : 100 %

Accuracy of  bird : 100 %

Accuracy of   cat : 75 %

Accuracy of  deer : 100 %

Accuracy of   dog : 66 %

Accuracy of  frog : 81 %

Accuracy of horse : 83 %

Accuracy of  ship : 91 %

Accuracy of truck : 100 %

## Misclassified Images

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/misclassified_images.png)

## Misclassified Images with Grad-CAM

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/misclassified_images_gradcam.png)

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/misclassified_images_gradcam1.png)
