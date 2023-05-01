import torch
from torch import nn
import torch.nn.functional as F



class ViT(nn.Module):
    def __init__(self, num_class, batch_size, patch_size, max_num_token, d_modle, n_head=8, n_block=4) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(batch_size, patch_size, d_modle, max_num_token)
        self.encoder_layer = nn.TransformerEncoderLayer(d_modle, n_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_block)
        self.linear = nn.Linear(d_modle, num_class)
        # self.patch_embedding.to(iter(self.linear.parameters()).__next__().device)

    

    def forward(self, X):
        # print("X.shape" , X.shape)
        x = self.patch_embedding(X) # 
        x = self.transformer_encoder(x)
        cls_token = x[:, 0, :]  # 取出cls_token对应的输出.
        print(cls_token.shape)
        return self.linear(cls_token)
        


class PatchEmbedding(nn.Module):
    def __init__(self, batch_size, patch_size, d_model, max_num_token, channel=3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.unfold = nn.Unfold(self.patch_size, stride=self.patch_size)
        # self.weight = torch.randn(size=((patch_size ** 2) * channel, d_model), requires_grad=True)
        self.linear = nn.Linear(patch_size*patch_size*channel, self.d_model, bias=False)
        self.cls_token_embadding = nn.Parameter(torch.zeros(size=(1, 1, self.d_model), requires_grad=True))

        self.position_embedding_table = nn.Parameter(torch.zeros(size=(max_num_token, d_model)))
        
        
    
    def forward(self, X):
        print(X.shape)
        print(self.cls_token_embadding.type)
        self.cls_token_embadding =  self.cls_token_embadding.expand(size=(X.shape[0], -1, -1)) # 将Tensor, 拓宽到指定大小, -1表示不改变维度. 

        patch = self.unfold(X).transpose(2, 1)
        # print("patch.shape", patch.shape)
        patch_embadding = self.linear(patch)
        print(self.cls_token_embadding.type)



        token_embadding = torch.cat([self.cls_token_embadding, patch_embadding], dim=1)
        token_embadding += self.position_embedding_table[: token_embadding.shape[1]]
        return token_embadding
    
    def init_weight(layer):
        pass




if __name__ == "__main__":
    # convert image to embedding embody. 
    images = torch.rand(size=(2, 3, 8, 8)).to(0)
    patch_size = 4
    D_vector = 8 # 就是每个patch用多少维度来表示. 这里起到demo作用.
    max_num_token = 16
    output = ViT(10, 2,  patch_size, 8, 128).to(0)
    # print(output(images))

    print(output(images).shape)











