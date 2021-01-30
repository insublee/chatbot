import math
import torch
import torch.nn as nn

from .encoder import CustomEncoder
from .wm import WorkingMemory
from .ltm import LongTermMemory
from .selector import Selector
from .decoder import CustomDecoder


class Chatbot(nn.Module):
    # step 1 : encoder-> TransformerEncoder -> memory -> TransformerDecoder -> decoder
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, d_memory, dropout=0.5):
        super(Chatbot, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformer_encoder = CustomEncoder(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, 
                 activation="relu", num_encoder_layers=nlayers, d_memory=d_memory)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.transformer_decoder = CustomDecoder(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, 
                 activation="relu", num_decoder_layers=nlayers, d_memory=d_memory)
        self.decoder = nn.Linear(ninp, ntoken)
        
        self.init_weights()
        
        
    def forward(self, src, tgt, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #print("src, src_mask :", src.size(), src_mask.size())
        memory = self.transformer_encoder(src, src_mask)
        
        
        tgt = self.encoder(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer_decoder(tgt, memory)
        output = self.decoder(output)
        return output
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
        """ 
    def forward(self, x, segment_info):
        
        sentence_vector = self.encoder(x, segment_info)
        retrieved_memories = self.long_term_memory(sentence_vector)
        current_memory = self.working_memory(sentence_vector)
        selected_memory = self.selector(retrieved_memories, current_memory)
        x = self.decoder(selected_memory)
        
        return x
    
    def encoder_decoder_forward(self,x, segment_info):
        sentence_vector = self.encoder(x, segment_info)
        x = self.decoder(sentence_vector)
        return x
        """
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)