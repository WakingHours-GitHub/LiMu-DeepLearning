"""
__call__: 函数: 就是将实例当作函数时候调用的方法.







常见的Transforms:
    我们需要关注这些Transform的就是输入, 输出, 以及作用.

    1. transforms.Compose([])
        将多个transforms函数组合, 依次执行, 使用[]将transform进行组合. 相当于一个容器.
        Compose()中的参数需要是一个列表, 然后列表中需要是transforms类的数据,
        因此是List([transforms, ...])
    2. transforms.ToTensor()
        将,PLT.image, ndarray类型的数据转换为Tensor类型.
    3. transforms.ToPILImage()
        将我们的Tensor对象转换为以一个PLT.image对象
    4. transforms.Normalize()
        归一化tensor类型的image, 使用mean和std对输入的tensor image每个通道进行归一化处理
        注意, mean, std应该和通道数同维度, 也就是有图片有几个维度, 那么mean和std就应该有对应维度.
        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    5. transforms.Resize()
        resize输入的PLT image 到给定的size大小, 注意, 只能使用PLT image类型, 不能使用其他类型
        如果给定是一个序列:(h, w)那么input就会缩放到这个大小
        如果只给定了一个数, 那么就按照最短边缩放到这个数值

        老版本只能使用Image对象, 新版本可以使用Tensor类型和Image类型.
        返回值就是输入类型, 如果输入Image, 返回就是Image, 如果输入Tensor同理, 返回Tensor
    6. RandomCrop()
        随机裁剪, 给定size, 进行裁剪. 最终返回结果.


    总结使用方法:
        关注transforms的输入和输出类型, 然后多看官方API.
        然后看源码, 需要什么参数.




"""
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np

writer = SummaryWriter("./logs")



def test03():
    """使用Compose()将多个transform进行组合. """

    # 我们将test02中的两部transform进行组合. 形成一个transforms对象, 这时我们就需要用到Compose

    trans_compose = transforms.Compose([ # 就是一个容器.
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor() # Conversation to Tensor type
    ])
    img_path = "./test_img.png"

    img_Image = Image.open(img_path)

    img_tensor = trans_compose(img_Image)

    writer.add_image("Compose", img_tensor)












def test02():
    """
    Resize()的使用:
    Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    重新缩放输入图片到给定大小, 如果图像是一个Tensor, 那么他是预期的.

    size (sequence or int): Desired output size. If size is a sequence like
    (h, w), output size will be matched to this. If size is an int,
    smaller edge of the image will be matched to this number.
    i.e, if height > width, then image will be rescaled to
    (size * height / width, size).

    其中, 这个size是预期的. 渴望一个输出size, 如果size是一个序列(sequence), 那么输出将会匹配这个尺寸.
    如果size是一个int类, 那么image最小的边将会匹配到这个数字上. 也就是进行一个等比例的一个缩放.


    """

    img_path = "./test_img.png"

    img = Image.open(img_path)
    print(img)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=774x753 at 0x7FC036062E60>

    # 使用Tensor:
    trans_totensor = transforms.ToTensor()  # 返回ToTensor对象

    img_tensor = trans_totensor(img)
    print(img_tensor.shape) # torch.Size([3, 512, 512])

    # 设置size.
    img_resize_sequence = transforms.Resize((512, 512))(img_tensor) # 直接生成实例对象, 然后调用call函数
    img_resize_int = transforms.Resize(512)(img_tensor)

    print(img_resize_sequence.shape) # torch.Size([3, 512, 512])



    writer.add_image("Resize", img_resize_sequence, 1)
    writer.add_image("Resize", img_resize_int, 2)










def test01():
    img_path = "./test_img.png"
    # use Image to read image
    img = Image.open(img_path)
    print(img)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=774x753 at 0x7FC036062E60>

    # 使用Tensor:
    trans_totensor = transforms.ToTensor()  # 返回ToTensor对象

    img_tensor = trans_totensor(img)

    # add to tensorboard
    writer.add_image("ToTensor", img_tensor)

    # Normalize: 归一化, 映射到一个范围中.
    trans_normalize = transforms.Normalize(
        mean=np.full(shape=(3,), fill_value=0.5),
        std=np.full(shape=(3,), fill_value=0.5)  # 注意, 是一维的
    )

    img_norm = trans_normalize(img_tensor)

    writer.add_image("img normalize", img_norm)


if __name__ == '__main__':
    # test01()
    # test02()
    test03()

writer.close()





