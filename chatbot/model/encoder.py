import torch
from typing import Tuple, Optional
from torch import Tensor
import torch.nn as nn



class CustomEncoder(nn.TransformerEncoder):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", num_encoder_layers=6, d_memory=2048):
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        super(CustomEncoder, self).__init__(encoder_layer, num_encoder_layers, encoder_norm)
        self.vectorizer = nn.Linear(d_model, d_memory)
    
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output) # S,N,E

        output = torch.mean(output, dim=0) # N,E
        output = self.vectorizer(output) # N, d_memory

        return output
