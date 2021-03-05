from typing import Tuple, Optional
from torch import Tensor
import torch


class CustomDecoder(torch.nn.Module):
    def __init__(
        self,
        decoder,
        WM_size, 
        LongTermMemory_size, 
        sentence_emb_dim, 
        max_seq_length,
        ):
        super().__init__()
        self.sentence_emb_dim = sentence_emb_dim
        self.max_seq_length = max_seq_length
        self.decoder = decoder

    def forward(self, encoded_WM, retrieved_action, action_label):
        # input : retrieved_action(batch, sentence_emb_dim)
        #         encoded_WM(batch, WM_size, sentence_emb_dim)
        # output: decoder_output(batch, max_seq_length, sentence_emb_dim)
        concated_feature = torch.cat((encoded_WM, retrieved_action.unsqueeze(1)), 1) # (batch, W.M.length + 1, sentence_emb_dim)

        decoder_output = self.decoder(input_ids = action_label['input_ids'],
                                 attention_mask = action_label['attention_mask'],
                                 encoder_hidden_states = concated_feature)
        return decoder_output # (batch, max_seq_length, sentence_emb_dim)