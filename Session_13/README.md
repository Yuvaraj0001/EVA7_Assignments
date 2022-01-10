# Vision Transformers with PyTorch

### (Submitted By Yuvaraj - Yuvaraj100493@gmail.com)

The objective is to train dogs and cats classification dataset using Vision Transformers. 

 Implemented the code using Cats & Dogs for Vision Transformers with PyTorch using vit_pytorch package and Linformer

## Dataset

Dataset is downloaded from Kaggle. 

The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog or cat (1 = dog, 0 = cat)

![alt_text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_13/Images/sample.png)

## Model

Model using vit-pytorch and Linformer https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session_13/ViT_CatsvsDogs.ipynb 

## Training Log

100%
313/313 [02:35<00:00, 2.32it/s]

Epoch : 16 - loss : 0.6007 - acc: 0.6686 - val_loss : 0.5861 - val_acc: 0.6806

100%
313/313 [02:32<00:00, 2.38it/s]

Epoch : 17 - loss : 0.5976 - acc: 0.6712 - val_loss : 0.5897 - val_acc: 0.6863

100%
313/313 [02:33<00:00, 2.39it/s]

Epoch : 18 - loss : 0.5934 - acc: 0.6785 - val_loss : 0.5839 - val_acc: 0.6897

100%
313/313 [02:32<00:00, 2.25it/s]

Epoch : 19 - loss : 0.5910 - acc: 0.6760 - val_loss : 0.5783 - val_acc: 0.6893

100%
313/313 [02:33<00:00, 2.34it/s]

Epoch : 20 - loss : 0.5871 - acc: 0.6840 - val_loss : 0.5785 - val_acc: 0.6879

## Reference

https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/

https://www.kaggle.com/tongpython/cat-and-dog
