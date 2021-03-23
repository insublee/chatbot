import torch.nn as nn
import torch

class SentimentAstimater(nn.Module):
    # 임시로 넣는거.
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x.mean(-1)
        x = x.mean(-1)
        return torch.abs(x.unsqueeze(-1))