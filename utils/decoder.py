####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    """
    Decoder layer with self and cross-attention
    """
    def __init__(self, dim, num_heads,mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
           nn.Linear(dim, int(mlp_ratio * dim)),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(int(mlp_ratio * dim), dim)
        )
    
    def forward(self, x, vision, causal_mask,return_att=False):
        # masked self-attention
        xl, _ = self.self_attn(self.norm1(x),self.norm1(x),self.norm1(x),attn_mask=causal_mask)
        x = x + xl
        
        # cross-attention
        att = []
        if return_att:
           # B, embeding_dim, 197 = <CLS> + 196 (if image 224, patch 16 => 14x14 = 196)
           xl, att = self.cross_attn(self.norm2(x), vision, vision, need_weights=return_att, average_attn_weights=False)
        else:
           xl, _ = self.cross_attn(self.norm2(x), vision, vision, need_weights=False, average_attn_weights=True)
        x = x + xl
        x = x + self.ffn(self.norm3(x))
        
        ret = (x, att) if return_att==True else x
        return ret

class TextEmbedding(nn.Module):
    """
    Prepare text tokens 
    """
    def __init__(self, vocab_size, dim, max_len=50):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)

    def forward(self, input_ids):
        _, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        return self.token(input_ids) + self.pos(pos)
