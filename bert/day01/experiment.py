#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from random import *
torch.manual_seed(4)
vocab_len = 10 #词表的大小
max_len = 20 #每句话的最大长度
batch_size = 4 #每个批次句子的数量
max_pred = 3 # 最多mask的token数量

sentence_len = torch.randint(1, max_len, (batch_size,)) #根据最大长度生成每个句子的具体长度
sentence = [torch.randint(4, vocab_len + 4, (L,)) for L in sentence_len] #根据每个句子具体长度生成token

#PAD : 0, CLS : 1, SEQ : 2, MASK : 3
negative = 0#二分类正样本
positive = 0#二分类负样本
batch = []
visit_seq = [] #保证不选重
while positive != batch_size / 2 or negative != batch_size / 2:
    a_index, b_index = randrange(len(sentence)), randrange(len(sentence))
    while (a_index, b_index) in visit_seq:#保证不选重
         a_index, b_index = randrange(len(sentence)), randrange(len(sentence))
    visit_seq.append((a_index, b_index))

    a, b = sentence[a_index], sentence[b_index]
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
    npad = max_len * 2 - len(input_ids)
    input_ids = torch.cat([input_ids, torch.zeros(npad,)]) #给这两个token进行padding
    segment_ids = torch.cat([segment_ids, torch.zeros(npad, )])
    if a_index + 1 == b_index and positive < batch_size / 2:#采样，最后一个bool标记是二分类任务的标记
        batch.append([input_ids, segment_ids, masked_pos, masked_tokens, True])
        positive += 1
    elif a_index + 1 != b_index and negative < batch_size / 2:
        batch.append([input_ids, segment_ids, masked_pos, masked_tokens, False])
        negative += 1



print(batch[])
