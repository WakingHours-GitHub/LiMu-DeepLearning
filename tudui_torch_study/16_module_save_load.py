"""
剩下的课程:
    模型保存与加载
    完成的训练套路
    实现使用GPU进行训练
    模型验证套路
    GitHub上面的优秀代码案例


模型的保存和模型的加载

模型的保存与加载
    两种保存方式: 模型的保存和加载一定要对应



# 陷阱:
注意, 方式1加载时有一个陷阱, 就是保存完模型后, 再加载时
需要再加载的文件中, 也定义出这个class这个模型, 但是你无需实例化它
然后才可以加载模型
我们也可以再开头中直接引入: from xxx import *
所以在使用方式1的时候, 加载时我们应该让文件能够访问到我们定义的这个模型(class中)
所以我们还是需要能够找到对应模型的class。需要模型的定义, 但是不用实例化. 


"""
import torch
import torchvision



def load_save_method2():
    vgg_net = torchvision.models.vgg16(pretrained=False)
    # 就是将模型加载进来的模型参数直接load进来为dict格式, 然后通过Module.load_state_dict()进行
    net_dict = torch.load("./16_module_save/vgg16_method2.pth")
    print(net_dict) # 打印参数字典。 OrderDict type
    vgg_net.load_state_dict(torch.load("./16_module_save/vgg16_method2.pth"))



    
def save_method2():
    """
    模型保存方式2, 
    只是保存模型中的参数, 所以需要现有网络实例, 然后load进来参数, 然后再load给实例.
    官方推荐, 因为这种保存方式只保存模型参数, 不保存网络模型结构. 所以保存空间比较小
    """
    vgg16 = torchvision.models.vgg16(pretrained=False) # 不加载预训练模型
    torch.save(vgg16.state_dict(), "16_module_save/vgg16_method2.pth")
    


def load_save_method1():
    vgg16 = torch.load("./16_module_save/vgg16_method1.pth")
    # 直接返回这个对象. 直接就是网络的实例.
    print(vgg16)

def save_method1():
    """
    使用方式1保存模型,
    这种方式又保存模型的参数部分, 又保存模型的网络结构, 
    加载时, 直接返回的就是一个模型对象
    """
    vgg16 = torchvision.models.vgg16(pretrained=False) # 不加载预训练模型
    torch.save(vgg16, "./16_module_save/vgg16_method1.pth")

def my_always_use_method():
    vgg16 = torchvision.models.vgg16(pretrained=False) # 不加载预训练模型
    torch.save(vgg16.state_dict(), "16_module_save/vgg16_dict.pth")

    


if __name__ == "__main__":
    save_method1()
    load_save_method1()
    my_always_use_method()





