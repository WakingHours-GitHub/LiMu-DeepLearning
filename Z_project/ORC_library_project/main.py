# from image_dataset import ImageDataset
import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn


writer = SummaryWriter("./logs")



EPOCH = 3
BATCH_SIZE = 64 
NUM_WORKERS = 4
# device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "CPU")
device = torch.device("cuda:0")
# print(device)


class ImageDataset(Dataset):
    def __init__(self, train=True, transforms=None, target_transfoms=None):
        self.root = "./image_data"
        self.train = True
        self.transforms = transforms
        self.target_transfoms = target_transfoms
        if train == True:
            self.root = os.path.join(self.root, "train")
        else: 
            self.root = os.path.join(self.root, 'test')
        self.file_name_list = os.listdir(self.root)

    def __getitem__(self, item):
        "item is index that "
        full_path = os.path.join(self.root, self.file_name_list[item])
        input = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
        label = self.file_name_list[item]
        if self.transforms != None:
            input = self.transforms(input)
        if self.target_transfoms != None:
            label = self.target_transfoms(label)
            label = label.reshape(label.shape[1:])

        return input, label

    def __len__(self):
        return len(self.file_name_list)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def label2one_hot(label) -> np.ndarray:
    label = np.array([int(ele) for ele in label.split(".")[0]])
    label = convert_to_one_hot(label, 10)
    return label



class OCR_netword(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(1, 4, (3, 3), 1, 0)
        self.pool1 = nn.MaxPool2d((3, 3), 1, 1, ceil_mode=True)
        self.relu1 = nn.ReLU()



    def forward(self, X):
        # X: torch.Size([64, 1, 50, 130])
        X = self.conv1(X)
        X = self.pool1(X)
        X = self.relu1(X)
        
        return X



def loss_function(y_hat, y_true):
    pass



def train() -> None:
    image_dataset = ImageDataset(
        transforms=transforms.ToTensor(), 
        target_transfoms=transforms.Compose([
            label2one_hot,
            transforms.ToTensor()
        ])
    )
    image_dataloader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    net = OCR_netword()

    for epoch in range(EPOCH):
        for idx, (images, labels) in enumerate(image_dataloader):
            # writer.add_images("source image", images, idx)

            images = images.cuda(0)
            print(images.shape)
            labels = labels.cuda()
            net = net.to(device)
            y_hat = net(images)
            print(y_hat.shape)
            print(labels.shape)
            # loss_function(y_hat, labels)



            
            break

        break










if __name__ == "__main__":
    train()
    pass



writer.close()
