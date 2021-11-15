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

4. Get Accuracy 85%.



## Result

1. No of Parameters - 99.85k

2. Highest Train Accuracy -70.49

3. Highest test Accuracy - 77.75

## Model

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%207/Images/model.JPG)
