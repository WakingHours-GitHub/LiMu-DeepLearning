from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2 as cv
 


class OCR_Dataset(Dataset):
    def __init__(self, train=True):
        self.root = "./image_data"
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

        return input, label.split('.')[0]

    def __len__(self):
        return len(self.file_name_list)




def main() -> None:
    dataset_picture = OCR_Dataset()
    print(dataset_picture[0][0].shape)



if __name__ == '__main__':
    main()