import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from models.layers import *
from models.transformer import *

B = 64
n = 50
d_model = 512
n_h = 8
d_k = d_model/n_h


t = Transformer(src_vocab_size=1000, tgt_vocab_size=2000, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)
source = torch.randn((B, n, d_model))
target = torch.randn((B, n, d_model)) 
t(source, target)

# multi_head = MultiHeadAttention(d_model, n_h)

# Q = torch.randn((B, n, d_model))
# K = Q.clone()
# V = Q.clone()

# multi_head(Q, K, V)
