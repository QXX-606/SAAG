import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size,time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.struct_emb_dim = 74
        #self.struct_emb = struct_emb
        self.norm = norm

        self.query_weight = torch.nn.Linear(self.struct_emb_dim, self.struct_emb_dim)
        self.key_weight = torch.nn.Linear(self.struct_emb_dim, self.struct_emb_dim)

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            #in_dims_temp是一个list，第一个元素是in_dims[0] + self.time_emb_dim，后面的元素是in_dims[1:]
            #in_dims_temp的size
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim ] + self.in_dims[1:]
            print(in_dims_temp)
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims[:-1] + [self.out_dims[-1] + self.time_emb_dim]
        out_dims_temp = self.out_dims
        print(out_dims_temp)
        # in_layers是一个ModuleList，包含了每一层的线性变换 
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def attention_count(self,x,struct_emb):
        # 使用嵌入矩阵1作为查询，嵌入矩阵2作为键
        queries = self.query_weight(x)
        keys = self.key_weight(struct_emb)
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # 使用softmax函数得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重计算加权的嵌入矩阵
        weighted_embed = torch.matmul(attention_probs, struct_emb)
        
        # 返回加权的嵌入矩阵
        return weighted_embed
    
    def forward(self, x, timesteps, struct_emb):
        
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(struct_emb.device)

        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        x = x.to(struct_emb.device)
        h = torch.cat([x, emb], dim=-1)
        print('h1',h.shape)
        h = self.attention_count(h,struct_emb)
        print('h2',h.shape)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        print('h3',h.shape)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        print('h4',h.shape)
        return h


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
