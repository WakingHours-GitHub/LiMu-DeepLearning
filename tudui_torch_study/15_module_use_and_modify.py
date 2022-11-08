"""

我们来学习一些经典的网络模型
我们学习一些经典的网络模型, 并且修改其中的一些参数

这次我们来介绍一个对图片继续分类的常用模型
torchvision.models.XXX下面就是一些已经提供好的模型。

VGG:
torchvision
def vgg16(
    pretrained: bool = False,  # 表示是否加载数据集imgNet和已经训练好的模型参数
    progress: bool = True # 是否加载下载进度条.
    ):

很多框架都会将VGG16作为一个前置的网络结构, 一般用VGG16提取一些感兴趣的特征
然后在加上自己的网络结构, 输出自己想要的结果.



"""
import torchvision
from torch import nn




def modify_vgg16_to_10_classes() -> None:
    """
    我们可以通过修改vgg16的模型, 将最后的1000类别, 塌陷到10类别当中去.

    """
    vgg16_false = torchvision.models.vgg16(pretrained=False) 
    vgg16_false.classifier.add_module("final_linear_to_10_classes_linear", nn.Linear(1000, 10)) #
    # 通过Module.add_module("name", Module)我们就可以添加名字, 以及对应的网络模型
    # 我们也可以直接添加到classifier这个sequential中去. 就是直接.对应的名字
    print(vgg16_false) 
    # 替换: 
    # 我们可以将Seqential看成一个列表(容器), 然后我们就可以通过索引来得到对应索引处的引用, 然后指向我们新的网络模型. 
    vgg16_false_replace = vgg16_false.classifier[-1] = nn.Linear(1000, 40)
    print(vgg16_false_replace)



def test() -> None: # 
    # ImageNet数据集有100多g, 非常大, 因此我们不推荐使用该数据集. 
    # train_data = torchvision.datasets.ImageNet("./data", split="train", download=True, 
    #             transform=torchvision.transforms.ToTensor())
    vgg16_false = torchvision.models.vgg16(pretrained=False) # 不加载网络模型的参数. 
    # vgg16_true = torchvision.models.vgg16(pretrained=True) # 加载网络模型参数. 

    print("ok") # 我们可以设置断电然后看看这两个网络模型中的参数到底有什么不同. 

    print(vgg16_false) # 我们可以看vgg16网络模型的参数.  
    # 最后输出的结果是1000, vgg16是在imagenet数据集上进行测试的. 所以输出的类别信息为1000
    





if __name__ == "__main__":
    # test()
    modify_vgg16_to_10_classes()










