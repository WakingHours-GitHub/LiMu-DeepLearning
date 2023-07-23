import torch, torchvision


from utils import get_train_vaild_datasets, load_data_CIFAR10, train_cos_ema


batch_size = 128

def test_ema_with_cos() -> None:
    train_iter, val_iter = load_data_CIFAR10(batch_size)
    


if __name__ == "__main__":
    test_ema_with_cos()
