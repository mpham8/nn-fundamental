import torch.nn as nn
import torch


class maskedSingleHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, maxTokens):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

        self.register_buffer('mask', torch.log(torch.tril(torch.ones(maxTokens, maxTokens)))) #buffer for attention mask so not recreated every forward pass


    def forward(self, x):
        tokens = x.shape[1] #x is batch x tokens x d_model

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        mask = self.mask[:tokens, :tokens]
        return torch.softmax ( (Q @ K.transpose(-2,-1)) / (self.d_k**0.5) + mask, dim=1) @ V  


class maskedMultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.heads = nn.ModuleList([maskedSingleHeadAttentionBlock(d_model, d_k, d_v) for i in range(h)]) #modulelist so can repeat
        self.W_o = nn.Linear(h * d_v, d_model)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim = -1)
        return self.W_o(x)

#decoder block
class decoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        #masked multihead attention

        #add and norm

        #multihead attention

        #add and norm

        #feed forward

        #add and norm