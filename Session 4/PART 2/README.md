# Neural Architecture

Buiding Architecture to achieve,
1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. Have used BN, Dropout, a Fully connected layer, have used GAP. 

### Model Architecture


![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%202/Images/Model_architect.JPG)

The model included Batch Normalization, Drop-out, Global Average Pooling. The total number of parameters are 187,146.

* `Batch Normalization`

We apply Batch normalization at every layer to normalize the inputs to the next layer.

* `Drop-out`
 
We apply drop out after batchnorm every time to regularize our input and to prevent our model from overfitting.

* `Global Average Pooling`

This layer is introduced when our channel dimension is 1x1. Not only does this translate convolutional structure to linear structure, it has the added advantage of having less parameters to compute and since it doesn't have to learn anything, it helps avoid overfitting. 

### Logs

Accuracy reached 99.4% at 8th epoch.

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%202/Images/logs.JPG)


### Conclusion

The architecture is built using batch normalization, drop out, fully connected layer and Global Average Pooling (GAP). 

The results attain 99.4% of accuracy within 20 epoch and number of parameters reached over 20k.

Need to learn more about GAP.

#### References:

The architecture is build based based on this link, https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99
