import torch
import torch.nn as nn

class CustomEncoder(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self,*input, **kwargs):
        return self.encoder(*input, **kwargs)

    def vectorizing(self, encoder_output, attention_mask):
        #First element of model_output contains all token embeddings
        token_embeddings = encoder_output[0] # batch, seq, hidden
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask