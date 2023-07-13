"""
在前面我们学习了dataset, dataset只是提供了一种方式, 将数据和对应的label读取进来. 并且建立索引.
dataloader, 是一个加载器, 也就是我们需要将数据加载到我们的神经网络当中,
我们现在使用的方法都是小批量SGD, 也就是我们不是将所有的数据一次性送入到网络中, 而是需要先进行batch一下.
dataloader就是随机从dataset取batch个数据, 然后送入到神经网络以供学习.



dataloader, datasets在torch.utils.data包下.
然后我们通过官网。查看其API。

torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False, pin_memory_device='')
    dataset (Dataset) – dataset from which to load the data.
    batch_size (int, optional) – how many samples per batch to load (default: 1).
    shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    sampler (Sampler or Iterable, optional) – defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.
    batch_sampler (Sampler or Iterable, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
    num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    collate_fn (callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
    pin_memory (bool, optional) – If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
    drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
    timeout (numeric, optional) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)
    worker_init_fn (callable, optional) – If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)
    generator (torch.Generator, optional) – If not None, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. (default: None)
    prefetch_factor (int, optional, keyword-only arg) – Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default: 2)
    persistent_workers (bool, optional) – If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: False)
    pin_memory_device (str, optional) – the data loader will copy Tensors into device pinned memory before returning them if pin_memory is set to true.

    介绍一下参数:
        dataset: 就是DataSet, 也就是之前定义好的我们自己的dataset实例传入进去即可.
        batch_size: 就是批量大小, 这是一个超参数.  默认为1.
        shuffle: 是否是shuffle也就是打乱. 默认为False.
        num_workers: load数据时, 使用的进程数. 默认0. 一般为2的次幂
        drop_last: 是否丢弃最后一个batch. 默认为False.
        sampler: 采样器, 就是以什么样的方式来组织数据.



"""



import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os, sys
os.chdir(sys.path[0])

write = SummaryWriter("./logs")

def use_dataloader() -> None:
    # 因为测试集比较小, 方便测试, 这里我们只是看一下效果
    test_dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=transforms.ToTensor(),
                                                download=True)

    img, label = test_dataset[0]
    print(img, label)

    # 然后我们生成DataLoader:
    test_dataloader = DataLoader( # 返回一个类迭代器iterator
        dataset=test_dataset,  # 选择对应的dataset
        batch_size=64,  # 超参数, 也就是一个批量的大小
        shuffle=True,  # 每次是否打乱
        num_workers=4,  # load data时候的线程数
        # drop_last=False,  # 是否要丢掉最后一个batch
        drop_last=True
    )
    for epoch in range(2):
        # 一轮epoch遍历的是全部batch数据
        # 我们设定了shuffle为True, 所以这两轮中, 每个采样的数据是不同的.

        # test_dataloader实际上是一个迭代器. (yield返回数据)
        for idx, data_batch in enumerate(test_dataloader):
            # print(idx)
            img_batch, label_batch = data_batch
            # print(img_batch.shape)  # torch.Size([64, 3, 32, 32])
            # print(label_batch.shape)  # torch.Size([64])

            # 我们可以进行一些展示:
            # 注意, 这里是add_images. 接受的参数是可以是4D的数据, 是批量的图片数据. Tensor对象, [B, C, H, W]
            # 而add_image, 只能接受3D的图片数据.
            write.add_images(f"DataLoader_{epoch}", img_batch, idx)
            """
            在TensorBoard中我们可以清晰的看到, 最后一个batch的数据. 是16张, 这是因为我们的总的数据数目, 无法整除batch_size
            所以导致最后一个batch是缺失的, 不过我们可以利用drop_last将最后一个batch进行丢掉. 
            
            """

        # break

if __name__ == '__main__':
    use_dataloader()



write.close()
