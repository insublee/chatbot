import torch
from typing import Tuple, Optional
from torch import Tensor
import torch.nn as nn


"""
class CustomEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder,
        WM_size : int, 
        LongTermMemory_size : int, 
        sentence_emb_dim : int, 
        max_seq_length : int,
        ):
        super().__init__()
        self.WM_size = WM_size
        self.sentence_emb_dim = sentence_emb_dim
        self.encoder = encoder
    
    def forward(self, x):
        
        #input_ids torch.Size([batch, dialogue, seq_len])
        #attention_mask torch.Size([batch, dialogue seq_len])
        #dialogue_mask torch.Size([batch, dialogue])
        
        
        first_tokens_size=torch.Size((x['input_ids'].size(0), self.WM_size + 2, self.sentence_emb_dim))
        first_tokens = torch.empty(first_tokens_size)
         # (batch, dialogue, hidden)
        
        action_label = {}
        for i in range(self.WM_size + 2): # for loop for dialogue length
            batch = {}
            for key in x.keys():
                if key != 'dialogue_mask':
                    # (batch, dialogue, seq_len).T(1,0) -> (dialogue, batch, seq_len)
                    # (dialogue, batch, seq_len)[i] -> (batch, seq_len))
                    batch[key]=x[key].transpose(1,0)[i].type(torch.LongTensor) # (batch,seq)
            output = self.encoder(**batch) # (batch,seq,sentence_emb_dim)
            # Allocate sentence vector (first vector of output last_hidden_state) for i th dialogue
            first_tokens[:,i] = output['last_hidden_state'][:,0,:].detach()
            
            if i == self.WM_size:
                action_label = batch
        
        return first_tokens, action_label # (batch, dialogue, hidden), (batch, seq)
"""

class CustomEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder,
        WM_size : int, 
        LongTermMemory_size : int, 
        sentence_emb_dim : int, 
        max_seq_length : int,
        ):
        super().__init__()
        self.WM_size = WM_size
        self.sentence_emb_dim = sentence_emb_dim
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        encoder는 첫번째로 wm 7개와 action을 각각 벡터라이즈 하면 됨. reward를 뽑을때 쓰는 next_state는 아마 밖에서 처리할듯.
        inputs : ['input_ids','attention_mask','segment_id'] or ['action_ids','action_attention_mask']
        (action은 LTM에 저장할때와 디코더에 저장할때만 쓴다. 여기에는 안들어옴.)
        
        return 
        """
        # working memory를 통째로 인코딩 
        encoder_out = self.encoder(**x)

        # vectorize
        if 'segment_id' in x.keys():
            encoded_wm, encoded_state = vectorizer(encoder_out, x['segment_id'])
            return encoded_wm, encoded_state
        else:
            return vectorizer(encoder_out)