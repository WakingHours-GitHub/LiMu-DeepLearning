# 人生苦短, 及时行乐. 生活也就那么回事, 没有你想象中的那么有意义.
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import ceil, dtype, ne, nn


writer = SummaryWriter("./logs")



EPOCH = 200
BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 0.05
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



class OCR_netword(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3, 3), 
            stride=1,
            padding=1,
            ceil_mode=True,
        )
        
        self.conv2=nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding=1,
            stride=2,
        )
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=(3, 3), 
            stride=1,
            padding=1,
            ceil_mode=True,
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8*25*65, 1024)
        self.finally_full_connection_layer = nn.Linear(1024, 40)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, X):
        # X: torch.Size([64, 1, 50, 130])
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)
        

        X = self.flatten(X)
        X = self.linear1(X)
        X = self.finally_full_connection_layer(X)

        # 神经网络输出结束. 然后我们应该对后续的一些形状进行处理,.
        X = X.reshape((-1, 4, 10))
        X = self.softmax(X) # 输出概率. 
        return X


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def label2one_hot(label) -> np.ndarray:
    label = np.array([int(ele) for ele in label.split(".")[0]])
    label = convert_to_one_hot(label, 10)
    return label


def calculate_accuracy(y_hat, y_true):
    y_true = y_true.reshape((-1, 4, 10))
    y_hat = y_hat.reshape(y_true.shape)

    y_hat_argmax = torch.argmax(y_hat, dim=-1)
    y_true_argmax = torch.argmax(y_true, dim=-1)
    accuracy = (y_hat_argmax == y_true_argmax).float().mean()
    # print(accuracy)
    return accuracy

    



def loss_function(y_hat, y_true):
    y_hat = y_hat.reshape(y_true.shape)
    # print(y_hat[0]) 
    # print(y_true.shape) # torch.Size([64, 4, 10])
    y_hat = y_hat.reshape((BATCH_SIZE, -1)).float()
    y_true = y_true.reshape((BATCH_SIZE, -1)).float()

    # y_hat = torch.argmax(y_hat, dim=-1).float()
    # y_true = torch.argmax(y_true, dim=-1).float()

    # print(y_hat)
    # print(y_true)

    loss = nn.MSELoss()
    loss = loss(y_hat, y_true)
    loss.to(device)

    return loss





def train(is_load=True) -> None:

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
        drop_last=True, # 丢掉最后一个batch.
    )

    net = OCR_netword()
    if is_load:
        net.load_state_dict(torch.load("./OCR_parameters.net"))
    net = net.to(device)


    trainer = torch.optim.SGD(net.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for idx, (images, labels) in enumerate(image_dataloader):
            # writer.add_images("source image", images, idx)

            images = images.cuda(0)
            labels = labels.cuda(0)

            y_hat = net(images)

            # print(y_hat.shape)
            # print(labels.shape) # torch.Size([64, 4, 10])

            trainer.zero_grad() # empty gradient. because will accumulate gradient in pytoch.
            loss = loss_function(y_hat, labels)
            loss.backward()
            trainer.step() # 更新参数.

        
        loss_value = loss_function(y_hat, labels)
        accuracy = calculate_accuracy(y_hat, labels)
        print(f"epoch: {epoch}: loss: {loss_value.item()}, accuracy: {accuracy.item()}")

        torch.save(net.state_dict(), "./OCR_parameters.net")


            



            
            # break

        # break


def test():
    
    root = "./image_data/train"
    file_name_list = os.listdir(root)
    file_full_path_list = [os.path.join(root, filename) for filename in file_name_list]

    


    choice_file_path_list = random.sample(file_full_path_list, 8)
    print(choice_file_path_list)

    _, figs = plt.subplots(2, 4, figsize=(16, 9))
    figs = figs.flatten() # 需要平坦.
    

    for idx, (f, file_path) in enumerate(zip(figs, choice_file_path_list)):
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        # image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        code = inference(image)
        print("code: ", code)
        print("file_path:", file_path)
        y_true = file_path.rsplit("/", -1)[-1].split('.')[0]
        
        f.imshow(image, cmap='gray')
        f.set_title(f"true: {y_true}\npred: {code}")
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)





        # break
    plt.savefig("./test.png")
    plt.show()
    




def inference(image):
    image = transforms.ToTensor()(image)
    image = image.reshape((1, )+image.shape)
    image = image.cuda(0)
    # print(image.shape) # torch.Size([1, 1, 50, 130])


    # 开始推理: 
    net = OCR_netword()
    net.load_state_dict(torch.load("./OCR_parameters.net"))
    net.eval() #  进行评估模式
    net = net.to(device)

    y_hat = net(image)
    y_hat = y_hat.reshape((4, 10))

    # print(y_hat)
    y_hat_argmax = torch.argmax(y_hat, dim=-1)
    # print(y_hat_argmax)
    # print(y_hat_argmax.tolist())

    code = "".join([str(ele) for ele in y_hat_argmax.tolist()])

    return code
    






if __name__ == "__main__":
    train(is_load=False)
    # test()
    pass



writer.close()
