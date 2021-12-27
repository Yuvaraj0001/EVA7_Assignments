# Yolov3 Object Detection on Custom Dataset

### Dataset Preparation

Dataset has 4 classes Hardhat, Mask, Vest, Boots. This repo is trained on this dataset and in addition to these images, 25 images for each class is downloaded and added to the above data.

### Data Annotation

Refer to this repo for annotating the images. File structure for image annotaion is as below.
yolov3

|── images
|   ├──img-001.jpg
|   ├──img-002.jpg
|   ├──...
├── classes.txt
├── main.py
├── process.py

run main.py to load the annotation tool and process.py to create test.txt and train.txt.

### Downloading pretrained weights

Download the file named yolov3-spp-ultralytics.pt and place it in the directory.

### Training Output

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_11/Custom/Images/img1.jpg)

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_11/Custom/Images/img2.jpg)

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_11/Custom/Images/img3.jpg)


### plots

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_11/Custom/Images/results.png)


```python

```
