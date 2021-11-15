# -*- coding: utf-8 -*-
"""dataloader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TZIVd0i4Uu9tZVg4_SVDQ1HUoKsPi3Qn
"""

import cv2
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                        fill_value=0.4734),
        A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
            p =1.0
        ),
        ToTensorV2()
    ],
    p=1.0
)

test_transforms = A.Compose(
    [
        A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
            p =1.0
        ),
        ToTensorV2()
    ]
)

class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 128
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 2, 'pin_memory': True} if self.use_cuda else {}

trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)
                                        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                          shuffle=True, **args(use_cuda=True).kwargs)


testset = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transforms)

testloader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                         shuffle=False, **args(use_cuda=True).kwargs)

##
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def get_statistics(dataset):

    """
    input = dataset should be a of type torchvision.datasets calculate statistics of training data only 
    Calculated stats are->
    torch.Size([3, 32, 32, 50000])
    Mean =  tensor([0.4914, 0.4822, 0.4465])
    Standard deviation =  tensor([0.2470, 0.2435, 0.2616])
    
    """
    imgs = torch.stack([img_t for img_t,_ in dataset],dim = 3)
    print(imgs.shape)

    mean_calc = imgs.view(3,-1).mean(dim = 1)
    std_dev_calc = imgs.view(3,-1).std(dim = 1)

    print('Mean = ',mean_calc)
    print('Standard deviation = ',std_dev_calc)

    return mean_calc,std_dev_calc