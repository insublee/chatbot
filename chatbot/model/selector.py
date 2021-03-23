import torch
import torch.nn.functional as F

class Selector(torch.nn.Module):
    def __init__(
        self,
        LongTermMemory_size,
        sentence_emb_dim,
        ):
        super().__init__()
        
        self.fc_pi_1 = torch.nn.Linear(sentence_emb_dim, sentence_emb_dim * 4)
        self.fc_pi_2 = torch.nn.Linear(sentence_emb_dim * 4, LongTermMemory_size)
        self.fc_v_1 = torch.nn.Linear(sentence_emb_dim, sentence_emb_dim * 4)
        self.fc_v_2 = torch.nn.Linear(sentence_emb_dim * 4, 1)

    def forward(self, retrieved_action):
        x = F.relu(self.fc_pi_1(retrieved_action)) # (batch, hidden*4)
        x = self.fc_pi_2(x) # (batch, LongTermMemory_size)
        memory_logits = F.softmax(x, dim=1)
        return memory_logits