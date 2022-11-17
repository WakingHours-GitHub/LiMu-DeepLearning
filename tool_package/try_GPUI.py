
import torch 

def try_GPU():
    torch.device("cuda:0" if torch.cuda.is_available() else "CPU")