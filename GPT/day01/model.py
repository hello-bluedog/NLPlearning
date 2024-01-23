#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def attention(Q, K, V, mask = None): # Q, K, V b * s * m
    d_k = Q.size(-1)
    score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, 1e-9)
    p_attn = F.softmax(score, dim = -1)
    return torch.matmul(p_attn, V), p_attn

def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class MultiheadedAttention(nn.Module):
    def __init__(self, h, model_dim, dropout = 0.1):
        super(MultiheadedAttention, self).__init__()
        assert model_dim % h == 0
        self.h = h
        self.d_k = model_dim // h
        self.linears = clones(nn.Linear(model_dim, model_dim), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):
        nbatches = Q.size(0)
        q, k, v = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for lin, x in zip(self.linears, [Q, K, V])]
        if mask is not None:
            mask = mask.unsqueeze(1)
        attn, _ = attention(q, k, v, mask, self.dropout)
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](attn)


class LayerNorm(nn.Module):
    def __init__(self, size, eps = 1e-9):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return (x - mean) / (self.eps + std) + self.b 

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedward(nn.Module):
    def __init__(self, model_dim, ffn, dropout = 0.1):
        super(PositionwiseFeedward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn)
        self.w2 = nn.Linear(ffn, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))


class DecodeLayer(nn.Module):
    def __init__(self, model_dim, attn, ffn, dropout = 0.1):
        super(DecodeLayer, self).__init__()
        self.sublayer = clones(SublayerConnection(model_dim, dropout), 2)
        self.attn = attn
        self.ffn = ffn
        self.size = model_dim

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, model_dim, vocab_size, max_len = 300):
        super(Embeddings, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_len, model_dim)
        self.norm = LayerNorm(model_dim)
    
    def forward(self, x):
        pos = self.arange(x.size(1))
        x = self.token_embed(x) + self.pos_embed(pos)
        return self.norm(x)
    
class Generator(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        prob = self.proj(x)
        return F.softmax(prob)


class GPT(nn.Module):
    def __init__(self, embed, decoder, generator):
        super(GPT, self).__init__()
        self.embed = embed
        self.decoder = decoder
        self.generator = generator

    def decode(self, x, mask):
        return self.decoder(x, mask)

    def forward(self, x, mask):
        logits = self.decode(x, mask)[-1]
        #prob = F.softmax(logits, dim = -1)[:-1]
        return prob

    def geedy_decode(self, x, max_len = 10):
        mask  = subsequent_mask(x.size(-1))
        ys = torch.tensor([])
        for i in range(max_len):
            out = self.decode(x, mask)
            prob = self.generator(out)[-1]
            _, next_token = torch.max(prob, dim = -1)
            ys = torch.zeros(1,1).fill(next_token.data[0])
            x = torch.cat([x, ys], dim = 1)
        return x


def make_model(vocab_size, model_dim = 512, ffn = 2048, h = 8, dropout = 0.1, N = 2):
    model = GPT(Embeddings(model_dim, vocab_size), Decoder(DecodeLayer(model_dim, MultiheadedAttention(h, model_dim), PositionwiseFeedward(model_dim, ffn)), N), Generator(model_dim, vocab_size))

        
make_model(10)
