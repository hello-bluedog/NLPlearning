#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# step1 数据准备
max_seq_len = 16
max_src_seq_len = 8
max_tgt_seq_len = 6
min_src_seq_len = 2
min_tgt_seq_len = 2
src_word_num = 8
tgt_word_num = 5
batch_size = 2

torch.manual_seed(4)
src_seq_len = torch.randint(min_src_seq_len, max_seq_len, (batch_size,))
tgt_seq_len = torch.randint(min_tgt_seq_len, max_seq_len, (batch_size,))

src_seq = torch.stack([F.pad(torch.randint(1, src_word_num, (L,)), (0, max_seq_len - L)) for L in src_seq_len], dim=0)
tgt_seq = torch.stack([F.pad(torch.randint(1, tgt_word_num, (L,)), (0, max_seq_len - L)) for L in tgt_seq_len], dim=0)
#print(tgt_seq)
#print(src_seq)

#step 2 word embedding
model_dim = 8
src_embedding_tbl = nn.Embedding(1 + max_seq_len, model_dim)
tgt_embedding_tbl = nn.Embedding(1 + max_seq_len, model_dim)
src_embedding = src_embedding_tbl(src_seq)
tgt_embedding = tgt_embedding_tbl(tgt_seq)
#print(src_embedding_tbl.weight)
#print(src_embedding)

#step 3 pos embedding
max_position = max_seq_len
pe = torch.zeros(max_position, model_dim)
pos_mat = torch.arange(max_seq_len).reshape(-1, 1)
div_term = torch.exp(torch.arange(0, model_dim, 2) * -math.log(10000)/ model_dim)
pe[:, 0::2] = torch.sin(pos_mat * div_term)
pe[:, 1::2] = torch.cos(pos_mat * div_term)
#print(pe)

#step 4 decoder casual mask
attn_shape = (1, max_seq_len, max_seq_len)
casual_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
print(casual_mask.shape)


#step 5 self-attn mask
self_attn = (src_seq != 0).unsqueeze(-2)
self_attn_mask = self_attn & self_attn.transpose(-2, -1)
#print(self_attn_mask)

# step 6 scaled-dot attention
def scaled_dot_attention(Q, K, V, mask = None):
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    print("---------in dot-----")
    print("score shape")
    print(scores.shape)
    print("mask shape")
    print(mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim = -1)
    return torch.matmul(p_attn, V)

self_attention = scaled_dot_attention(src_embedding, src_embedding, src_embedding, self_attn_mask)
#print(self_attention)
#print(self_attention.shape)
infra_attention = scaled_dot_attention(tgt_embedding, src_embedding, src_embedding, self_attn_mask)
#print(self_attention)
#print(infra_attention.shape)

# step 7 multi head scaled-dot attention

def multi_head_attention(Q, K, V, mask = None, h = 2, d_model = 8):
    if mask is not None:
        mask = mask.unsqueeze(1)
    d_k = d_model // h
    nbatches = Q.size(0)
    q, k, v = [ x.view(nbatches, -1, h, d_k).transpose(1, 2) for x in (Q, K, V)]
    x = scaled_dot_attention(q, k, v, mask)
    x = x.transpose(1, 2).contiguous().view(nbatches, -1, d_k * h)
    return x
m_atten = multi_head_attention(src_embedding, src_embedding, src_embedding, self_attn_mask, 2, 8)
