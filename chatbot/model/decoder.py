from typing import Tuple, Optional
from torch import Tensor
import torch.nn as nn

class CustomDecoder(nn.TransformerDecoder):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", num_decoder_layers=6, d_memory=2048):
        
        #decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_layer = CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, d_memory)
        decoder_norm = nn.LayerNorm(d_model)

        super(CustomDecoder, self).__init__(decoder_layer, num_decoder_layers, decoder_norm)
        

class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, d_memory):
        super(CustomDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=d_memory, vdim=d_memory)