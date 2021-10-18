# Neural Architecture

Buiding Architecture to achieve,
1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. Have used BN, Dropout, a Fully connected layer, have used GAP. 

### Model Architecture


![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%202/Images/Model_architect.JPG)

The model included Batch Normalization, Drop-out, Global Average Pooling. The total number of parameters are 187,146.

### Logs

Accuracy reached 99.4% at 8th epoch.

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%202/Images/logs.JPG)


### Conclusion

The architecture is built using batch normalization, drop out, fully connected layer and Global Average Pooling (GAP). 

The results attain 99.4% of accuracy within 20 epoch and number of parameters reached over 20k.

Need to learn more about GAP.

#### References:

The architecture is build based based on this link, https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99
