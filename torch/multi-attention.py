import numpy as np
from typing import Optional, List
import torch 
from torch import nn 
# import pytorch_lightning as pl 

from labml import tracker

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        # d_model: input dimension
        # heads: number of heads
        # d_k: the dimension of each head, the number of features per head
        super().__init__()
        self.linear = nn.Linear(d_model, heads*d_k, bias=bias) # embedding matrix
        self.heads = heads
        self.d_k = d_k 

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch_size, d_model] or [batch_size, d_model]
        head_shape = x.shape[:-1] 
        x = self.linear(x) # linear transform, W^Qx, W^Kx, W^Vx
        x = x.view(*head_shape, self.heads, self.d_k) 
        # output: [seq_len, batch_size, heads, d_k] or [batch_size, d_model]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads 
        self.heads = heads 
        
        # Query matrix, Key matrix, Value Matrix
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias) 
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / np.sqrt(self._dk)
        self.attn = None 
        
    def get_atnn_score(self, query: torch.Tensor, key: torch.Tensor):
        """compute attention score:  QK^T
        Q: [seq_len, batch_size, heads, d_k]
        K: [seq_len, batch_size, heads, d_k]
        """
        # using einsum
        # [seq_len, d_k] [d_k, seq_len]
        # keep batch_size b and heads h 
        return torch.einsum('ibhd, jbhd->ijbh', query, key)  # i bh d, j bh d -> i bh j -> i j bh
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0] 
        assert mask.shape[1] == key_shape[0]    
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        
        # same mask applied to all heads
        mask = mask.unsqueeze(-1) # [seq_len_q, seq_len_k, batch_size, heads]
        return mask  

    def forward(self, *, 
                query: torch.Tensor,  # [seq_len, batch_size, d_model]
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None): # [seq_len, seq_len, batch_size]
        # mask[i,j,b] : whether for batch b , query at position i has access to key-value at position j .
        seq_len, batch_size, _ = query.shape 
        
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value) 
        
        # compute attention scores, and scale scores
        scores = self.get_atnn_score(query, key)
        scores *= self.scale  

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # use -inf to replace the value where its mask=0 (True)
        
        attn = self.softmax(scores)  # softmax(attn_score)
        tracker.add({'attention': attn}) # add monitoring for attention
        # apply dropout 
        attn = self.dropout(attn)
        # multiply by values
        # [seq_len, seq_len] [seq_len, d_k]
        x = torch.einsum('ijbh, jbhd->ibhd', attn, value) # i j bh, j bh d -> i bh d 
        
        self.attn = attn.detach() # save attentions for any other calculations
        # concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1) . # i b h d -> i b h*d
        return self.output(x)  # final linear transform

        