# ----- Written by Ahmad Farhan ------

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''

    query, key, value : All are projections of input
    mask : To ensure future words are unreachable
    '''
    B, _, T, d_k = query.size()
    scores = torch.matmul(query, key.transpose(-2, -1)) / \
        (math.sqrt(d_k))  # dot product b/w query,key

    if mask is not None:
        # make future words unreachable -inf
        scores = scores.masked_fill(mask[:, :, :T, :T] == 0, -1e10)
    prob_attn = F.softmax(scores, dim=-1)  # calculating probs
    if dropout is not None:
        prob_attn = dropout(prob_attn)  # pass through dropout

    # attn_weights * value # weighted sum of values. each emb idx is a weighted sum of all other emb idxs of all T values
    return torch.matmul(prob_attn, value)


class CausalSelfAttention(nn.Module):

    ''' 

        n_head : number of attention heads
        block_size : context length
        attn_pdrop : attention dropout probability
        resid_pdrop : dropout prob after projection layer.

    '''

    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0  # d_model/n_head are divisble
        self.d_k = d_model//self.n_head

        # key, value, query, out_proj
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # to hide future words
        # subsequent_mask = torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size)
        # self.register_buffer("mask",subsequent_mask) # to make sure it is stored in states dict while saving model

    def forward(self, x, mask):
        B, T, d_model = x.size()
        query, key, value = [l(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (x, x, x))]
        # print(x.shape)
        y = attention(query, key, value, mask=mask, dropout=self.attn_drop)
        y = y.transpose(1, 2).contiguous().view(B, T, d_model)
        # print(y.shape)
        # pass through a linear and dropout
        return self.resid_drop(self.linears[-1](y))
