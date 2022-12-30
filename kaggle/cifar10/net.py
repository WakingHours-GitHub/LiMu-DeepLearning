import torch
from torch import nn
from torch.nn import functional as F


# 实现snet: 通道注意力的典型表现：
class senet(nn.Module):
    def __init__(self, channel, ratio=16) -> None:
        # radio用于表示第一个全连接后的缩放比例。
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # 输出高宽
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(), # 最后使结果塌陷到0,1之间. 
        )


    def forward(self, X):
        b, c, h, w = X.size()
        avg = self.avg_pool(X).reshape((b, c))
        fc = self.fc(avg).reshape((b, c, 1, 1))

        return X * fc # 对应元素相乘. 


# CBAM: 是通道注意力和空间注意力机制的一个结合: 
# 通道注意力机制
class channel_atteentaion(nn.Module):
    def __init__(self, channle, ratio=16) -> None:
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1) # 全局最大池化。 #
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化。 

        self.fc = nn.Sequential(
            nn.Linear(channle, channle//ratio, False),
            nn.ReLU(),
            nn.Linear(channle//ratio, channle, False)
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor):
        b, c, h, w = X.size()

        max_pool_out = self.max_pool(X)
        avg_pool_out = self.avg_pool(X)

        # fc共享参数
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out) 

        max_fc_out += avg_fc_out
        out = self.sigmoid(max_fc_out).reshape([b, c, 1, 1])

        return out * X

# 实现空间注意力机制
class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2,1 ,kernel_size, stride=1, padding=kernel_size//2) # 不改变形状的conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        max_pool_out = torch.max(X, dim=1, keepdim=True)
        avg_pool_out = torch.mean(X, dim=1, keepdim=True)

        # concat:
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.conv(pool_out) #  二维
        # 然后经过sigmoid塌陷到(0, 1)之间
        out = self.sigmoid(out)

        return out * X



class CBAM(nn.Module):
    def __init__(self,  channle, ratio=16, kernel_size=7) -> None:
        super().__init__()
        self.channel_attenation = channel_atteentaion(channle, ratio)
        self.spacial_attention = spacial_attention(kernel_size)

    def forward(self, X):
        x = self.channel_attenation(X)
        x = self.spacial_attention(x)
        return x




class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, is_use1x1Conv=False, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if is_use1x1Conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=stride)
        else:
            self.conv3 = None
        

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)


        Y += X
        
        return F.relu(Y)

def res_stage(in_channels, out_channels, num_residuals, is_first=False):
    block = []

    for i in range(num_residuals):
        if i == 0 and not is_first: # 第一个stage不做高宽减半. 
            block.append(Residual(in_channels, out_channels, True, 2))
        else: 
            block.append(Residual(out_channels, out_channels)) # 后面之后就不进行通道的增加了。
    
    return block

    

class resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    

        self.stage1 = nn.Sequential(
            nn.Conv2d(3,  64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=1, padding=1),
            
        )

        self.stage2 = nn.Sequential(*res_stage(64, 64, 2, is_first=True)) # 第一个不做高宽减半. 
        
        self.stage3 = nn.Sequential(*res_stage(64, 128, 2))
        self.stage4 = nn.Sequential(*res_stage(128, 256, 2))
        self.stage5 = nn.Sequential(*res_stage(256, 512, 2))

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 10)

        self.init_parameters() # 调用一下

    def forward(self, X):
        X = self.stage1(X)


        X = self.stage2(X)
        X = self.stage3(X)
        X = self.stage4(X)
        X = self.stage5(X)

        X = self.global_avg(X)
        X = self.flatten(X)
        X = self.linear(X)
        return X
        

    def test(self):
        X = torch.rand(size=(1, 3, 32, 32))
        # X = self(X)
        # print(X.shape)

        for layer in self._modules:
            X = eval("self."+layer)(X)
            print(layer, X.shape)


    def init_parameters(self):
        print("init parameters used by xavier_uniform method. ")
        def init_weight(layer):
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        self.apply(init_weight)
        
    



