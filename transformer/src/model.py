import torch.nn as nn
import torch

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


        

class attentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)


    def forward(self, x):
        tokens = x.shape[1] #x is batch x tokens x d_model

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        mask = torch.log (torch.tril(torch.ones(tokens, tokens)))
        torch.softmax ( (Q @ K.transpose(-2,-1)) / (self.d_k**0.5) + mask, dim=1) @ V  