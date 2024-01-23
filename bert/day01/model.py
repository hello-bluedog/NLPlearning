#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from random import *
import math

def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

def attention(Q, K, V, mask = None, dropout = None): # mat scale mask softmax
    d_k = Q.size(-1)
    score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score.masked_fill_(mask == 0, 1e-9)
    p_attn = F.softmax(score, dim = -1)
    if dropout is not None:
        score = dropout(score)
    return  torch.matmul(score, V), p_attn


class bert(nn.Module):
    def __init__(self, encoder, embed, generator, task1, task2):
        super(bert, self).__init__()
        self.encoder = encoder
        self.clsTask = task1
        self.maskTask = task2
        self.embed = embed
        self.generator = generator
    
    def inference(self, x, seg, mask):
        out = self.embed(x, seg)
        return self.encoder(out, mask)
    
    def forward(self, x, seg, mask, masked_pos):
        out = self.inference(x, seg, mask)
        cls_logit = self.clsTask(out)
        lm_logit = self.maskTask(out, masked_pos)
        return lm_logit, cls_logit


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, size, eps = 1e-9):
        super(LayerNorm, self).__init__()

        self.w = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.w * (x - mean) / (std + self.eps) + self.b

class MultiheadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiheadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None, dropout = None):
        nbatch = Q.size(0)
        q, k, v =  [lin(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (Q, K, V))]
        if mask is not None:
            mask = mask.unsqueeze(1)
        atten, _ = attention(q, k, v, mask, self.dropout)
        atten = atten.transpose(1, 2).contiguous().view(nbatch, -1, self.h * self.d_k)
        if dropout is not None:
            atten = self.dropout(atten)
        del q
        del k
        del v
        return self.linears[-1](atten)


class PositionwisedFeedback(nn.Module):
    def __init__(self, dff, model_dim, dropout = 0.1):
        super(PositionwisedFeedback, self).__init__()
        self.w1 = nn.Linear(model_dim, dff)
        self.w2 = nn.Linear(dff, model_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncodeLayer(nn.Module):
    def __init__(self, attn, model_dim, dff, dropout = 0.1):
        super(EncodeLayer, self).__init__()
        self.attn = attn
        self.ffn = dff
        self.sublayers = clones(SublayerConnection(model_dim, dropout), 2)
        self.size = model_dim

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x : self.attn(x, x, x, mask))
        x = self.sublayers[1](x, self.ffn)
        return x

class Embeddings(nn.Module):
    def __init__(self, vocab_size,  model_dim, dropout = 0.1, max_seq_len = 5000):
        super(Embeddings, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)
        self.seg_embed = nn.Embedding(2, model_dim)
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(model_dim)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len).unsqueeze(0).expand_as(x)
        x = self.token_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.dropout(self.norm(x))

class Generator(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        prob = self.proj(x)
        return F.softmax(prob, dim = -1)

class clsTask(nn.Module):
    def __init__(self, model_dim, dropout = 0.1):
        super(clsTask, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(model_dim, 2)
            )
    def forward(self, x):
        logits = self.cls(x[:,0])
        return logits

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class maskTask(nn.Module):
    def __init__(self, model_dim, generator):
        super(maskTask, self).__init__()
        self.active = gelu
        self.linear = nn.Linear(model_dim, model_dim)
        self.generator = generator
        self.model_dim = model_dim

    def forward(self, x, masked_pos):
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, self.model_dim)
        pred = self.active(self.linear(torch.gather(x, 1, masked_pos)))
        logits = self.generator(pred)
        return logits


def make_model(vocab_size, model_dim = 512, dff = 2048, h = 8, N = 2):
    c = copy.deepcopy
    attn = MultiheadedAttention(h, model_dim)
    ffn = PositionwisedFeedback(dff, model_dim)
    embed = Embeddings(vocab_size,  model_dim)
    generator = Generator(model_dim, vocab_size)

    model = bert(Encoder(EncodeLayer(c(attn), model_dim, c(ffn), ), N), c(embed), c(generator), clsTask(model_dim), maskTask(model_dim, c(generator)))

    return model

#make_model(11)



def get_std_mask(x, pad = 0):#x :  1 * s
    mask = (x != pad).unsqueeze(0)
    zeroIndex = torch.nonzero(x == pad).squeeze()
    mask = mask.expand((x.size(0), x.size(0))).index_fill(0, zeroIndex, 0)
    return mask
def test_model():
    model = make_model(24)
    torch.manual_seed(4)
    batch_size = 4
    max_pred = 3
    max_len = 10
    vocab_len = 20
    seq_len = torch.randint(1, max_len, (batch_size,))
    seq = [torch.randint(4, vocab_len + 4, (L,)) for L in seq_len]
    #mask = get_std_mask(seq[0], 0)
    negative = 0#二分类正样本
    positive = 0#二分类负样本
    batch = []
    visit_seq = [] #保证不选重
    while positive != batch_size / 2 or negative != batch_size / 2:
        a_index, b_index = randrange(len(seq)), randrange(len(seq))
        while (a_index, b_index) in visit_seq:#保证不选重
             a_index, b_index = randrange(len(seq)), randrange(len(seq))
        visit_seq.append((a_index, b_index))

        a, b = seq[a_index], seq[b_index]
        #0 pad mask 2 cls 1 seq 3
        input_ids = torch.cat([torch.ones(1,), a, torch.ones(1,).fill_(2), b, torch.ones(1,).fill_(2)])#预测任务token
        segment_ids = torch.cat([torch.zeros(2 + len(a)), torch.ones(1 + len(b))])#二分类token

        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))#该句子中需要预测的token数量
        can_masked_pos = [i for i, token in enumerate(input_ids) if token != 1 and token != 2]
        pos_masked = sample(can_masked_pos, n_pred)#随机选masked的位置
    # print(pos_masked)

        masked_tokens = []
        masked_pos = []
        for pos in pos_masked:
            masked_pos.append(pos)
            p = random()
            if p < 0.8:#以0.8的概率masked掉
                input_ids[pos] = 3
                masked_tokens.append(3)
            elif p > 0.9:#以0.1的概率随机选一个token填上
                index = randint(4, vocab_len + 4)
                input_ids[pos] = index
                masked_tokens.append(index)
            #以0.1的概率不变
        if max_pred > n_pred: #如果能够masked掉的token数量不够，那么填上pad
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        npad = max_len * 2 + 3 - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(npad,)]).to(torch.int32) #给这两个token进行padding
        segment_ids = torch.cat([segment_ids, torch.zeros(npad, )]).to(torch.int32)
        masked_pos = torch.tensor(masked_pos)
        if a_index + 1 == b_index and positive < batch_size / 2:#采样，最后一个bool标记是二分类任务的标记
            batch.append([input_ids, segment_ids, masked_pos, masked_tokens, True])
            positive += 1
        elif a_index + 1 != b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_pos, masked_tokens, False])
            negative += 1
    mask = get_std_mask(batch[0][0])
    #print(mask.unsqueeze(0).shape)
    logits = model.inference(batch[0][0].unsqueeze(0), batch[0][1].unsqueeze(0), mask.unsqueeze(0))
    lm_logits, cls_logits = model.forward(batch[0][0].unsqueeze(0), batch[0][1].unsqueeze(0), mask.unsqueeze(0), batch[0][2].unsqueeze(0))
    print(lm_logits, cls_logits)

test_model()
