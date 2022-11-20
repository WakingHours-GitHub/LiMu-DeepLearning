
import torch 

# def try_GPU():
#     torch.device("cuda:0" if torch.cuda.is_available() else "CPU")


def try_GPU(index=0):
    if torch.cuda.device_count() >= index+1:
        return torch.device(f"cuda:{index}")
    return torch.divide("cpu")

def try_all_GPU():
    "return all available GPUs, or [cpu(), if no GPU exists"
    devices = [torch.device(f"cuda:{i}") for i in torch.cuda.device_count()]
    return devices if devices else [torch.device("cpu")]