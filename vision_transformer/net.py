import torch
from torch import nn
import torch.nn.functional as F



def init_parameters(self):
    print("init parameters used by xavier_uniform method. ")
    def init_weight(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
    self.apply(init_weight)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, d_model, num_class, channel=3, nhead=8, num_layers=4) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, d_model, channel=channel)
        self.cls_to_token = nn.Identity() # 什么操作也不做, 一致.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, num_class)
        



    def forward(self, x):
        embedding = self.patch_embedding(x)
        x = self.transformer_encoder(embedding)
        cls_token = self.cls_to_token(x[:, 0, :]) # 取出cls token对应的transformer的输出
        return self.linear(cls_token) # 从d_model映射到类别的分布



class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, d_model, channel=3) -> None:
        super().__init__()
        assert image_size/patch_size == image_size//patch_size, "images_size has be divisible with patch_size"

        self.unfold = nn.Unfold(patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size*patch_size*channel, d_model)
        self.cls_token_embedding = nn.Parameter(torch.randn((1, 1, d_model), requires_grad=True), requires_grad=True)
        self.position_embedding_table = nn.Parameter(torch.randn(1, (image_size//patch_size)**2+1, d_model, requires_grad=True))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x:torch.Tensor = self.unfold(x).transpose(1, -1)
        x = self.linear(x) # each patch represented a vector with D length.
        # x = self.norm(x)


        cls_token_embedding = self.cls_token_embedding.expand(x.shape[0], -1, -1) # 坑.
        # expand返回的只是一个view(视图), 我们需要用新的指针指向这个视图, 如果使用原来的, 那么仍然是Parameter对象. 不是Tensor.
        x = torch.cat([cls_token_embedding, x], dim=1) + self.position_embedding_table

        return x




def test():
    device = [torch.device("cuda:0"),torch.device("cuda:1"),]

    images = torch.randn(size=(2, 3, 32, 32)).to(0)
    # patchembedding = torch.nn.DataParallel(PatchEmbedding(images.shape[-1], 8, 128), device).to(0)
    # embedding = patchembedding(images)
    net = nn.DataParallel(ViT(32, 8, 128, 10), device).to(0)
    output = net(images)
    
    print(output.shape)


if __name__ == "__main__":
    test()
