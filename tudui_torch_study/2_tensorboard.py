"""
Tensorboard的使用, 我们可以通过Tensorboard中查看loss的变化, 也可也通过观察loss来选择合适的超参数.


help:
    Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.

    将event文件写入log_dir中, event files能够通过TensorBoard进行解析.
    构造函数:
        def __init__(
        self,
        log_dir=None, # event文件写入的地址.
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ):


"""
# 导入SummaryWriter类.
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import numpy as np
# help(SummaryWriter)

# 创建对象
writer = SummaryWriter("./logs")

# 我们会使用这两个函数:
image_path = "./hymenoptera_data/train/ants/0013035.jpg"
image = cv.imread(image_path) # 读取进来直接就是ndarray类型.
print(image.shape) # (512, 768, 3) -> (H, W, C)
# 那么add_image,默认是(3, H, W) 我们需要设置指定形式即可.
writer.add_image("ants", image, 1, dataformats="HWC")
# 同一个tag, 会显示多张. 那么我们也可以通过滑动bar来控制显示图片的大小, 这里bar的数值就对应参数中的global_step参数.
"""
    # Add image data to summary.
        def add_image(
        self, 
        tag, # title, 标题
        img_tensor, # 需要: Tensor, ndarray类型的图像数据, 一般我们就是用cv读入即可, 读取的图像直接为ndarray类型.
        global_step=None, # 仍然是步骤. 
        walltime=None, 
        dataformats="CHW"
    ):
    
            Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
    
            图片张量: 默认是(3, H, W), 如果图像形状是(H, W, 3)那么我们需要设置dataformats这个参数为: HWC.
    

"""


for i in range(100):
    writer.add_scalar("y=sqrt(x)", i ** 0.5, i)

# writer.add_scalar()

"""
    # Add scalar data to summary.
    def add_scalar( 
        self,
        tag, # 标题, title
        scalar_value, # 对应的数值, 也就是y
        global_step=None, # 多少步, 也就是x轴.
        walltime=None,
        new_style=False,
        double_precision=False,
    ):

    tensorboard --logdir==文件夹名字 # 这样我们就可以打开tensorboard了
    我们也可以指定端口打开: -> --port=端口. 
    tensorboard --logdir= --port=端口号


"""

writer.close()  # 关闭
