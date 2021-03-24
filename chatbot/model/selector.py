import torch
import torch.nn.functional as F

class Selector(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fc_pi_1 = torch.nn.Linear(hparams.sentence_emb_dim, hparams.sentence_emb_dim * 4)
        self.fc_pi_2 = torch.nn.Linear(hparams.sentence_emb_dim * 4, hparams.LongTermMemory_size)
        self.fc_v_1 = torch.nn.Linear(hparams.sentence_emb_dim, hparams.sentence_emb_dim * 4)
        self.fc_v_2 = torch.nn.Linear(hparams.sentence_emb_dim * 4, 1)

    def forward(self, state_vector):
        x = F.relu(self.fc_pi_1(state_vector)) # (batch, hidden*4)
        x = self.fc_pi_2(x) # (batch, LongTermMemory_size)
        memory_logits = F.softmax(x, dim=1)
        return memory_logits