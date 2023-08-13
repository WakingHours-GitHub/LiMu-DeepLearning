from torch.utils.data import DataLoader
import torchvision
import sys
import os
from torchvision import transforms
from imgaug import augmenters as iaa
import numpy as np

    
os.chdir(sys.path[0])

from utils import *
from EfficientNet import *
img_size = 224





sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#     sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.


class iaa_transform:
    def __init__(self) -> None:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)    
        self.seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        #随机裁剪图片边长比例的0~0.1
        iaa.Crop(percent=(0,0.1)),
        #Sometimes是指指针对50%的图片做处理
        iaa.Sometimes(
            0.5,
            #高斯模糊
            iaa.GaussianBlur(sigma=(0,0.5))
        ),
        #增强或减弱图片的对比度
        iaa.LinearContrast((0.75,1.5)),
        #添加高斯噪声
        #对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
        #剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
        #改变像素点的颜色(不仅仅是亮度)
        iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),
        #让一些图片变的更亮,一些图片变得更暗
        #对20%的图片,针对通道进行处理
        #剩下的图片,针对图片进行处理
        iaa.Multiply((0.8,1.2),per_channel=0.2),
        #仿射变换
        iaa.Affine(
            #缩放变换
            scale={"x":(0.8,1.2),"y":(0.8,1.2)},
            #平移变换
            translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
            #旋转
            rotate=(-25,25),
            #剪切
            shear=(-8,8)
        )
    #使用随机组合上面的数据增强来处理图片
    ],random_order=True)

    def __call__(self, img):
        img = np.array(img)
        return torch.Tensor(self.seq(images =img))
    
    
    
train_transforme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.RandomResizedCrop(img_size, scale=(0.60, 1.0), ratio=(1.0, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    # iaa_transform(),

    transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                         [0.2023, 0.1994, 0.2010])
])

val_transforme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size, antialias=True),
    transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                         [0.2023, 0.1994, 0.2010])

])

cifar10_train_datasets = torchvision.datasets.CIFAR10(
    "~/data/torch_datasets", train=True, download=True, transform=train_transforme)
cifar10_val_datasets = torchvision.datasets.CIFAR10(
    "~/data/torch_datasets", train=False, download=True, transform=val_transforme)


lr = 0.054
batch_size = 256
num_epoch = 200


def train_val_with_cos() -> None:
    # gen_cifar10_train_val_text("cifar_data")
    train_iter = DataLoader(cifar10_train_datasets, batch_size, True, num_workers=25)
    val_iter = DataLoader(cifar10_val_datasets, batch_size, False, num_workers=25)
    

    net = efficientnet_b0(10)
    train_cos_ema(
        net, nn.CrossEntropyLoss(), 
        train_iter, val_iter,
        lr, num_epoch,
        save_mode="best", test_epoch=1,
        # load_path="/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar100/runs/exp5/weights/epoch4_testacc0.0512_loss0.035_acc0.032.pth"
        devices=try_indexs_gpus([4, 5, 6, 7])
    )


def test_submission() -> None:
    net = efficientnet_b0(10)
    
    test_to_submission(
        net, 
        "/home/wakinghours/programming/LiMu-DeepLearning/kaggle/cifar10_version2/logs/epoch98_testacc0.908_loss0.0035_acc0.92.pth"
    )

if __name__ == "__main__":
    train_val_with_cos()
    
    # test_submission()
