"""
通过查看官网API: 


"""
import cv2 as cv  # 用来读取图片.
import numpy
import os

import torch
from torch.utils.data import Dataset, DataLoader

# print(help(Dataset))

import sys
os.chdir(sys.path[0])


class myDataSet(Dataset):
    def __init__(self, load_path: str, label_dir: str, is_train=True) -> None:
        super().__init__()
        self.root_dir = os.path.join(load_path, f"{'train' if is_train else 'val'}")
        self.file_name_list = os.listdir(os.path.join(self.root_dir, label_dir))
        self.label_dir = label_dir

    def __getitem__(self, index):
        img = cv.imread(os.path.join(self.root_dir, self.label_dir, self.file_name_list[index])) # HWC
        print(img.shape)
        # 理论上来说, img还需要经过transform
        return img, self.label_dir
    
    def __len__(self):
        return len(self.file_name_list)


def main() -> None:
    print(myDataSet("../data/hymenoptera_data/", "ants")[0])


if __name__ == "__main__":
    main()
