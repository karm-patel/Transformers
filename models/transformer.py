import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from models.layers import *

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_seq_length, dropout, src_pad_token_val, tar_pad_token_val, device=torch.device("cuda")):
        super(Transformer, self).__init__()
        self.src_pad_token_val = src_pad_token_val
        self.tar_pad_token_val = tar_pad_token_val
        self.device = device

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != self.src_pad_token_val).unsqueeze(1).unsqueeze(2) # (B, 1, 1, n, d_model)
        tgt_mask = (tgt != self.tar_pad_token_val).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # generate lower-triangular matrix, upper-values will be 1e-9 in masking layers
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output