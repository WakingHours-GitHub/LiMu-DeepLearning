import torch
from torch import nn
import torch.nn.functional as F
from typing import *
# from vision_transformer.net_before import ViT
from net import *
from utils import *
import sys
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# os.environ["all_proxy"] = "http://172.31.179.129:15732"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6"
torch.cuda.device_count()
os.chdir(sys.path[0])

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# devices = try_all_GPUS() 
# print(devices)

BATCH_SIZE = 512
LR = 1e-2
EPOCH = 500

def main() -> None:

    # train_datasets = torchvision.datasets.CIFAR10(
    #     "../pytorch/data", True, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
            
    #     ]),
    #     download=True
    # )
    # test_datasets = torchvision.datasets.CIFAR10(
    #     "../pytorch/data", False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
            
    #     ])
    # )
    # train_iter = DataLoader(train_datasets, BATCH_SIZE, True)
    # test_iter = DataLoader(test_datasets, BATCH_SIZE, True)


    # train_iter, test_iter = load_CIFAR10_iter(BATCH_SIZE)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                     [0.2023, 0.1994, 0.2010])
            ])
    val_transform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Normalize([0.4914, 0.4822, 0.4465],  # normalize. 归一化.
                                     [0.2023, 0.1994, 0.2010])
            ])
    train_datasets = torchvision.datasets.CIFAR100("/data/fanweijia/data/torchvision_datasets", True,transform, download=True)
    val_datasets = torchvision.datasets.CIFAR100("/data/fanweijia/data/torchvision_datasets", False,val_transform, download=True)
    train_iter = DataLoader(
        train_datasets, BATCH_SIZE, True, num_workers=10
    )
    test_iter = DataLoader(
        val_datasets, BATCH_SIZE, True, num_workers=10
    )
    # print(iter(train_iter).__next__())
    # f, l = iter(train_iter).__next__()



    net = ViT(32, 8, 768, 100, nhead=16, num_layers=8)
    # output = net(f)
    # output = nn.Softmax()(output)
    # print(output.argmax(dim=1), l)
    # print(output.argmax(dim=1) == l)
    # print(output)
    
    # exit(0)
    
    # evaluate_test_with_GPUS(net, test_iter)
    # train_cifar10(net, LR, BATCH_SIZE, EPOCH)

    train_cos(
        net, nn.CrossEntropyLoss(),
        train_iter, test_iter,
        LR, EPOCH

    )
    # train(
    #     net, nn.CrossEntropyLoss(),
    #     train_iter, test_iter,
    #     LR, EPOCH, 10, 0.95, 5e-3
    # )


def image2embedding_naive(images, patch_size, weight):
    # images: 
    # image to patch的过程, 其实就是一个卷积, 然后stride为k, 然后每次我们取出卷积窗口中的应用
    # api: torch.nn.functional.unfold(input, kernel_size, ..) 就是跟卷积一样的参数, 然后可以取出卷积中的内容.
    # 输入必须是4D的东西, 返回: [batch, 滑动窗口的大小, 滑动的次数(也就是多少个patch)]
    patch = F.unfold(images,  patch_size, stride=patch_size).transpose(2, 1) # (Batch, patch_size*patch_size*channel, patch_num)
    # patch = F.unfold(images,  patch_size, stride=patch_size).transpose(-1, -2) # (Batch, patch_size*patch_size*channel, patch_num)

    # 变换维度. 

    # 然后将patch_size*patch_size*channel也就是每个patch中通道数的像素数目拉成一条向量. 然后transpose到后面的维度, 进行矩阵乘法, 映射到D
    # 也就是使用D(一个固定长度的向量)表示每个patch. 每个D中就是表示patch的语义信息.
    patch_embedding = patch @ weight
    # print(patch_embedding.shape) # (Batch, patch_num, D)
    
    return patch_embedding 






def image2embedding_conv():
    pass     






def test():
    # convert image to embedding embody. 
    images = torch.rand(size=(1, 3, 8, 8))
    patch_size = 4
    D_vector = 8 # 就是每个patch用多少维度来表示. 这里起到demo作用.
    max_num_token = 16
    patch_depth = patch_size ** 2 * 3 # patch*patch*channel, 实际上就是每一块的像素点. 
    weight = torch.rand(size=(patch_depth, D_vector))


    patch_embedding = image2embedding_naive(images, patch_size, weight) # (batch, patch_num, D)




    # cls token embedding:
    # random initialize
    cls_token_embedding = torch.rand(size=(1, 1, D_vector), requires_grad=True) # 是一个可以训练的参数, 所以我们需要梯度.

    # concat patch and cls token. is finally
    token_embedding = torch.cat([cls_token_embedding, patch_embedding], dim=1) # 在patch_num这个维度进行concat. 
    # 注意要记住cls_token的位置. 

    # position embedding, that is a can learning parameter also. 
    # size is (max_num_token, D)
    position_embedding_table = torch.randn(size=(max_num_token, D_vector), requires_grad=True)
    seq_len = token_embedding.shape[1] # 看一个token有多少个patch个数

    token_embedding+=position_embedding_table[: seq_len] # broadcast operate.

    # transformer encoder: transform block don't change shape of input, this feature make it can construct a lot of block. 
    encoder_layer = nn.TransformerEncoderLayer(D_vector, nhead=8) # transformer block.
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6) # encoder 
    encoder_output = transformer_encoder(token_embedding)

    # print(encoder_output.shape) # == input shape

    # get output: cls_token to classification via MLP.
    cls_token = encoder_output[:, 0, :] # (batch, patch, D) # 第一个path就是cls对应的patch所以取出.
    linear_layer = nn.Linear(D_vector, 10)
    output = linear_layer(cls_token)
    
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, torch.tensor([1]))
    print(loss.item())




if __name__ == "__main__":
    # test()
    main()
