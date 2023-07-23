import torch
from torch import nn
import torch.nn.functional as F


class CBAP(nn.Module): # Conv2d BatchNorm2d, Activate, MaxPool
    def __init__(self, in_channels, out_channels, ) -> None:
        super().__init__()
    

class VanillaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv4x4 = nn.Conv2d(3, 32, 4, 1, 0)
        # self.act1 = activation(32)
        self.conv1_1 = nn.Conv2d(32, 64, 1, 1, 1)
        
        self.conv1_2 = nn.Conv2d(64, 128, 1, 1, 1)        
        
    def forward()




class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = self.create_parameter(shape=(dim, 1, act_num*2 + 1, act_num*2 + 1), default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.bias = None
        self.bn = nn.BatchNorm2D(dim, epsilon=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return F.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num*2 + 1)//2, groups=self.dim)
        else:
            return self.bn(F.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn.epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = self.create_parameter(shape=[self.dim], default_initializer=nn.initializer.Constant(0.0))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True
        
        
        
        
class Block(nn.Layer):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2D(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2D(dim, dim, kernel_size=1),
                nn.BatchNorm2D(dim, epsilon=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2D(dim, dim_out, kernel_size=1),
                nn.BatchNorm2D(dim_out, epsilon=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2D(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2D((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x,self.act_learn)
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn.epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight = kernel
        self.conv1[0].bias = bias
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight = torch.matmul(kernel.transpose([0, 3, 2, 1]), self.conv1[0].weight.squeeze(3).squeeze(2)).transpose([0, 3, 2, 1])
        self.conv.bias = bias + (self.conv1[0].bias.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True
class VanillaNet(nn.Layer):
    def __init__(self, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768],
                 drop_rate=0, act_num=3, strides=[2,2,2,1], deploy=False, ada_pool=None, **kwargs):
        super().__init__()
        self.deploy = deploy
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2D(in_chans, dims[0], kernel_size=4, stride=4),
                activation(dims[0], act_num)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2D(in_chans, dims[0], kernel_size=4, stride=4),
                nn.BatchNorm2D(dims[0], epsilon=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2D(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2D(dims[0], epsilon=1e-6),
                activation(dims[0], act_num)
            )

        self.act_learn = 1

        self.stages = nn.LayerList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy)
            else:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy, ada_pool=ada_pool[i])
            self.stages.append(stage)
        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2D((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2D(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.AdaptiveAvgPool2D((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2D(dims[-1], num_classes, 1),
                nn.BatchNorm2D(num_classes, epsilon=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2D(num_classes, num_classes, 1)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        tn = nn.initializer.TruncatedNormal(std=.02)
        zero = nn.initializer.Constant(0.0)
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            tn(m.weight)
            zero(m.bias)

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = F.leaky_relu(x,self.act_learn)
            x = self.stem2(x)

        for i in range(self.depth):
            x = self.stages[i](x)

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            x = F.leaky_relu(x,self.act_learn)
            x = self.cls2(x)
        return x.reshape((x.shape[0],-1))

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn.epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight = kernel
        self.stem1[0].bias = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight = paddle.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight)
        self.stem1[0].bias = bias + (self.stem1[0].bias.reshape((1,-1,1,1)) * kernel).sum(3).sum(2).sum(1)
        self.stem = nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight = kernel
        self.cls1[2].bias = bias
        kernel, bias = self.cls2[0].weight, self.cls2[0].bias
        self.cls1[2].weight = paddle.matmul(kernel.transpose([0, 3, 2, 1]), self.cls1[2].weight.squeeze(3).squeeze(2)).transpose([0, 3, 2, 1])
        self.cls1[2].bias = bias + (self.cls1[2].bias.reshape((1,-1,1,1)) * kernel).sum(3).sum(2).sum(1)
        self.cls = nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True
def vanillanet_5(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], **kwargs)
    return model

def vanillanet_6(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], **kwargs)
    return model

def vanillanet_7(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,2,1], **kwargs)
    return model

def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,2,1], **kwargs)
    return model

def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,1,2,1], **kwargs)
    return model

def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,2,1],
        **kwargs)
    return model

def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,2,1],
        **kwargs)
    return model

def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,2,1],
        **kwargs)
    return model

def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    return model

def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    return model

def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        ada_pool=[0,40,20,0,0,0,0,0,0,10,0],
        **kwargs)
    return model
