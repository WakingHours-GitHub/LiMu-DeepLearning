"""
Tensor是什么.


transforms就是主要用作对图片进行一些变换。
我们需要使用transforms.


导入:
    from torchvision import transforms
transform的结果和用法: 直接查看transforms的源代码. 里面都有非常详细的说明.
我们常常用到的有:
    Compose() # 就是将多个transform对象进行组合. 相当于一个容器.
    ToTensor() # 将给定的Image和ndarray转换成Tensor类型.
    Resize() # 将图像重新缩放.

看一下思路是什么:
    就是转换器. 我们给定一些图片, 然后经过transform最终得到我们想要的结果.

魔术方法: __call__(): 就是将实例(对象)当作函数调用时, 默认调用该方法. 参数列表也是相同.

我们需要先生成实例, 然后使用实例函数, 进行转换. 这样做是为了Compose组合.
只有通过这种方法. 我们才能...
我们可以将很多个transform进行组合, 最后返回组合后的对象, 然后调用__call__方法, 最后调用, 完成实现.

我们可以看一下Tensor对象的attribute.
    backward_hooks: 就是反向传播的钩子, 锚点
    grad: 梯度
    grad_fn: 函数
    data: 数据
    dtype: 属性
    device: 设备






"""
import torch
from torchvision import transforms  # 导入transforms.
from PIL import Image # 通过Image读取的就是Image类型.
import cv2 as cv # 通过opencv读取进来的就是ndarray
from torch.utils.tensorboard import SummaryWriter # TensorBoard



# 代码:
# 首先我们通过Image读取进来一张Image图片
img_PIT = Image.open("hymenoptera_data/train/ants/0013035.jpg")
print(img_PIT) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x7F061900FAC0>

# 这显然不符合我们的Tensor数据类型. 所以我们需要对其使用transform进行转换.
# 生成transform实例
trans_obj = transforms.ToTensor() # 返回对象
tensor_img = trans_obj(img_PIT) # 将对象当作函数调用, 默认调用__call__()魔术方法
 #
print(type(tensor_img)) # <class 'torch.Tensor'> # 这就是Tenosr对象.


writer = SummaryWriter("logs")




# 使用opencv读取:
img_ndarray = cv.imread("hymenoptera_data/train/ants/0013035.jpg")
img_tensor = trans_obj(img_ndarray)
print(type(img_tensor)) # <class 'torch.Tensor'>

writer.add_image(
    tag="Tensor",
    img_tensor=img_tensor,
    global_step=2
)

writer.close()









