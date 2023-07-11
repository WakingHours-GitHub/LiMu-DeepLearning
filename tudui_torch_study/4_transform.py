import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2 as cv

import os, sys

os.chdir(sys.path[0])


writer = SummaryWriter("./logs")




def main() -> None:
    img = cv.imread("./test_img.png")
    print(img.shape)
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((416, 416))
    ])(img)
    
    print(type(img_tensor))
    
    writer.add_image("test_img_tensor iamge", img_tensor, 1)
    
    
    

if __name__ == "__main__":
    main()





writer.close()

