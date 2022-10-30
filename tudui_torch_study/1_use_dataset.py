"""
想要理解Dataset, 还需要多练习才可以,
我们需要做的, 就是重写getitem方法, 他返回一个元组, 一个是feature, 一个是label
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv  # cv读取进来的图片格式, 底层直接就是ndarray, 可以直接转换成Tensor
from PIL import Image  # 而Image读取进来的是PIL中的类. 我们还需要进行转换,

class MyData(Dataset):
    # 实现自己的数据集, 我们主要是重写__getitem__这个函数
    def __init__(self, root_path: str, label: str, is_train: bool = True):
        """
        初始化一些后面用得到的路径. 提供全局变量.
        """
        self.root_path = root_path
        self.label = label
        self.path = os.path.join(self.root_path, "train" if is_train else "val", self.label)  # 可以传入可变参数.
        self.file_name_list = os.listdir(self.path)

        pass

    def __getitem__(self, item):
        """
        这里item就是索引, 用于取出数据
        返回input, label.
        注意我们实现getitem, 返回值一定是input, label
        """
        feature = cv.imread(os.path.join(self.path, self.file_name_list[item]))
        label = self.label

        return feature, label #

    def __len__(self):
        """
        返回该数据集的长度.
        """
        return len(self.file_name_list)


def main() -> None:
    # 测试, 创建
    root = "./hymenoptera_data"
    ants_label = "ants"
    bees_label = 'bees'
    ants_dataset = MyData(root, ants_label)
    bees_dataset = MyData(root, bees_label)

    # print(ants_dataset[0]) # (feature, label)类型.
    # print(bees_dataset[0]) # (feature, label)类型.
    print(len(ants_dataset))  # 124
    print(len(bees_dataset))  # 121

    # 将两个Dataset合并成为一个Dataset:
    train_dataset = ants_dataset + bees_dataset  # 直接使用+ # 即可将两个Dataset组合起来
    print(type(train_dataset))  # <class 'torch.utils.data.dataset.ConcatDataset'> # concat: 合并多个数组.
    print(len(train_dataset))  # 245 # 也就是说我们将两个数据集相加起来.

    print(train_dataset[123])  # 蚂蚁
    print(train_dataset[124])  # 蜜蜂


if __name__ == '__main__':
    main()
