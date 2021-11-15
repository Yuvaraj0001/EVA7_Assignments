# Advanced Concepts

1. Run this network.  

2. Fix the network above:
    
    1. change the code such that it uses GPU
    
    2. change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
    
    3. total RF must be more than 44
    
    4. one of the layers must use Depthwise Separable Convolution
    
    5. one of the layers must use Dilated Convolution
    
    6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    
    7. use albumentation library and apply:
        
        horizontal flip
        
        shiftScaleRotate
        
        coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    
    8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k. 
    
    9. upload to Github
    
    10. Attempt S7-Assignment Solution. Questions in the Assignment QnA are:
        
        1. Which assignment are you submitting? (early/late)
        
        2. Please mention the name of your partners who are submitting EXACTLY the same assignment. Please note if the assignments are different, then all the names mentioned here will get the lowest score. So please check with your team if they are submitting even a slightly different assignment. 
        
        3. copy paste your model code from your model.py file (full code) [125]
        
        4. copy paste output of torchsummary [125]

        5. copy-paste the code where you implemented albumentation transformation for all three transformations [125]

        6. copy paste your training log (you must be running validation/text after each Epoch [125]

        7. Share the link for your README.md file. [200]

## Target

1. Use one of the layer Depthwise Separable Convolution and Dilated Convolution.

2. Use Albumentation

3. Set the architecture to reduce the parameters under 200k.

4. Get Consitent Accuracy 85%.



## Result

1. No of Parameters - 99.85k

2. Highest Train Accuracy -70.49

3. Highest test Accuracy - 77.75

## Model
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 16, 16]           2,080
             ReLU-10           [-1, 32, 16, 16]               0
           Conv2d-11           [-1, 64, 16, 16]          18,432
             ReLU-12           [-1, 64, 16, 16]               0
      BatchNorm2d-13           [-1, 64, 16, 16]             128
          Dropout-14           [-1, 64, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]             576
           Conv2d-16           [-1, 32, 18, 18]           1,024
             ReLU-17           [-1, 32, 18, 18]               0
      BatchNorm2d-18           [-1, 32, 18, 18]              64
        Dropout2d-19           [-1, 32, 18, 18]               0
           Conv2d-20             [-1, 32, 9, 9]           1,056
             ReLU-21             [-1, 32, 9, 9]               0
           Conv2d-22             [-1, 64, 7, 7]          18,432
             ReLU-23             [-1, 64, 7, 7]               0
      BatchNorm2d-24             [-1, 64, 7, 7]             128
          Dropout-25             [-1, 64, 7, 7]               0
           Conv2d-26             [-1, 32, 7, 7]          18,432
             ReLU-27             [-1, 32, 7, 7]               0
      BatchNorm2d-28             [-1, 32, 7, 7]              64
          Dropout-29             [-1, 32, 7, 7]               0
           Conv2d-30             [-1, 32, 4, 4]           1,056
             ReLU-31             [-1, 32, 4, 4]               0
           Conv2d-32             [-1, 32, 4, 4]           9,216
             ReLU-33             [-1, 32, 4, 4]               0
      BatchNorm2d-34             [-1, 32, 4, 4]              64
          Dropout-35             [-1, 32, 4, 4]               0
           Conv2d-36             [-1, 32, 4, 4]           9,216
             ReLU-37             [-1, 32, 4, 4]               0
      BatchNorm2d-38             [-1, 32, 4, 4]              64
          Dropout-39             [-1, 32, 4, 4]               0
AdaptiveAvgPool2d-40             [-1, 32, 1, 1]               0
           Linear-41                   [-1, 10]             330
================================================================
Total params: 99,850
Trainable params: 99,850
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.23
Params size (MB): 0.38
Estimated Total Size (MB): 4.62
----------------------------------------------------------------