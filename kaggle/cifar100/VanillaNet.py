import torch
from torch import nn
import torch.nn.functional as F


class CBAP(nn.Module): # Conv2d BatchNorm2d, Activate, MaxPool
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, act=F.leaky_relu, pooling=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.BN = nn.BatchNorm2d(out_channels)
        self.act = act
        self.pooling = nn.MaxPool2d(2) if pooling else nn.Identity()
        
    
    def forward(self, x) -> torch.Tensor:
        x = self.pooling(self.act(self.BN(self.conv(x))))
        return x
        
    

class VanillaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv4x4 = CBAP(3, 64, 5, 1)
        # self.conv4x4_2 = CBAP(32, 64, 3, 1)
        # self.act1 = activation(32)
        self.conv1_1 = CBAP(64, 128, 1, 1, 0, pooling=True)
        self.conv1_2 = CBAP(128, 256, 1, 1, 0, pooling=True)
        
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1)) # (7, 7) -> (1, 1)
        
        self.flatten = nn.Flatten()
    
        self.linear1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, 10)
        
        
        self.apply(self.init_net)
        
    def forward(self, x):
        x = self.conv1_2(self.conv1_1(self.conv4x4(x)))
        # x = x + y
        x = self.global_pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x
    
    def init_net(self, layer: nn.Module):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    

def try_net(X: torch.Tensor, net: nn.Module):
    net.to(X.device)

    for layer in net:
        X = layer(X)
        print(f"{layer.__class__.__name__:.10}: \t{X.shape}")



if __name__ == "__main__":
    x = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    print(VanillaNet()(x).shape)
    
    # try_net(torch.zeros((1, 3, 32, 32), dtype=torch.float32), VanillaNet())
