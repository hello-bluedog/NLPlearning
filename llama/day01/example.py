#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
#RoPE

def precompute_freqs_cis(dim : int, seq_len : int , theta : float = 10000.0):
    freq_cis = 1 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(seq_len)

    freq_cis = torch.outer(t, freq_cis).float()
    return torch.polar(torch.ones_like(freq_cis), freq_cis)


def rotray_embed(xq, xv, freq_cis):
    xq_ = xq.reshape(xq.size(0), xq.size(1), -1, 2).float()
    xv_ = xq.reshape(xv.size(0), xv.size(1), -1, 2).float()
    xq_ = torch.view_as_complex(xq_)
    xv_ = torch.view_as_complex(xv_)

    xq = torch.view_as_real(xq_ * freq_cis).flatten(2)
    xv = torch.view_as_real(xv_ * freq_cis).flatten(2)
    return xq, xv

#seq1 = torch.randint(0, 9, (3,4,4))
#seq2 = torch.randint(0, 9, (3,4,4))

#freq = precompute_freqs_cis(4, 4)
#emb = rotray_embed(seq1, seq2, freq)
#print(emb[0].shape)
#print(emb)
