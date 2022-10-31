from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2 as cv
from torchvision import transforms
import numpy as np

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
    

def main() -> None:
    dataset_picture = ImageDataset(
        transforms=transforms.ToTensor(), 
        target_transfoms=transforms.Compose([
            label2one_hot,
            transforms.ToTensor()
        ])
    )
    print(dataset_picture[0])



if __name__ == '__main__':
    main()