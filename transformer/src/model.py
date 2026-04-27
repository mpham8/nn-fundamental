import torch.nn as nn
import torch


class MaskedSingleHeadAttentionBlock(nn.Module):
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


class MaskedMultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.heads = nn.ModuleList([MaskedSingleHeadAttentionBlock(d_model, d_k, d_v) for i in range(h)]) #modulelist so can repeat
        self.W_o = nn.Linear(h * d_v, d_model)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim = -1)
        return self.W_o(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.sig1 = nn.ReLU()
        self.fc2 = nn.Linear(d_model*4, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig1(x)
        return self.fc2(x)


#decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, maxTokens):
        super().__init__()
        self.attn = MaskedMultiHeadAttentionBlock(d_model, d_k, d_v, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x)) #masked multihead attention + add/norm
        return self.norm2(x + self.ff(x)) #feed forward + add/norm


#entire transformer
class Transformer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, maxTokens, nx, vocabSize):
        super().__init__()        
        self.tokenEmbedding = nn.Embedding(vocabSize, d_model)
        self.positionalEmbedding = nn.Embedding(maxTokens, d_model)
        self.decoderBlocks = nn.ModuleList([DecoderBlock(d_model, d_k, d_v, h, maxTokens) for i in range(nx)])
        self.lin = nn.Linear(d_model, vocabSize)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device = x.device) #x.shape[1] gets seq len, torch.arange() generaties tensor of position idx, 

        x = self.tokenEmbedding(x) + self.positionalEmbedding(positions)
        for db in self.decoderBlocks: x = db(x)
        return self.lin(x)
