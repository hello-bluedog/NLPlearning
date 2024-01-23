#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import math

def clones(modules, N):
    return nn.ModuleList([copy.deepcopy(modules) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0

def attention(Q, K, V, mask = None, dropout = None):
    d_k = Q.size(-1)
    score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    p_attn = score.softmax(dim = -1)
    if(dropout):
        p_attn = dropout(p_attn)
    return torch.matmul(score, V), p_attn



# encoder 架构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src_seq, src_mask):
        return self.encoder(self.src_embed(src_seq), src_mask)

    def decode(self, memory, src_mask, tgt_seq, tgt_mask):
        return self.decoder(self.tgt_embed(tgt_seq), memory, src_mask, tgt_mask)

    def forward(self, src_seq, tgt_seq, self_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt_seq, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.softmax(self.proj(x), dim = -1)

class LayerNorm(nn.Module):
    def __init__(self, feature_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(feature_shape))
        self.b = nn.Parameter(torch.zeros(feature_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.norm = LayerNorm(layer.size)
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#residual module

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.ffn)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecodeLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, ffn, dropout):
        super(DecodeLayer, self).__init__()
        self.size = size
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.ffn = ffn
        self.self_attn = self_attn
        self.src_attn = src_attn

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.ffn)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, f"d_model is {d_model}, h is {h}"
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.linear = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, Q, K, V, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = Q.size(0)
        q, k, v = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for lin, x in zip(self.linear, (Q, K, V))]
        x, self_attn = attention(q, k, v, mask = mask, dropout = self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del q
        del k
        del v
        return self.linear[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, dff)
        self.w2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        print(x.shape)
        print(self.pe[:, :x.size(1)].shape)
        x =  x + self.pe[:, :x.size(1)].requires_grad_(False)   
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, dff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(d_model, h, dropout)
    ff = PositionwiseFeedForward(d_model, dff, dropout)
    position = PositionEmbedding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), #?为啥size是一个数
        Decoder(DecodeLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model



#--------------------testing----------------
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


#def run_tests():
#    for _ in range(10):
#        inference_test()
#run_tests()



