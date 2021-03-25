import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentAstimater(nn.Module):
    def __init__(self, hparams):

        super().__init__()
        self.hparams = hparams

        self.embedding = nn.Embedding(
            self.hparams.vocab_size,
            self.hparams.EMBEDDING_DIM,
            padding_idx = self.hparams.padding_index,
            )

        self.embedding.weight.data[self.hparams.padding_index] = torch.zeros(self.hparams.EMBEDDING_DIM)

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1, 
                out_channels = self.hparams.N_FILTERS, 
                kernel_size = (fs, self.hparams.EMBEDDING_DIM),
                ) 
            for fs in self.hparams.FILTER_SIZES
            ])

        self.dropout = nn.Dropout(self.hparams.DROPOUT)

        self.fc = nn.Linear(len(self.hparams.FILTER_SIZES) * self.hparams.N_FILTERS, self.hparams.OUTPUT_DIM)
        self.m = nn.Sigmoid()

    def forward(self, text):
        '''
        text = [batch_size, sentence_length]
        embedded = [batch_size, 1, sentence_length, embedding_dimesion]
        conved_n = [batch_size, N_FILTERS, sentence_length - FILTER_SIZES[n] + 1]
        pooled_n = [batch_size, N_FILTERS]
        cat = [batch_size, N_FILTERS * len(FILTER_SIZES)]
        '''

        embedded = self.embedding(text).unsqueeze(1)
        conved_n = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled_n = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_n]
        cat = self.dropout(torch.cat(pooled_n, dim = 1))
        return F.sigmoid(self.fc(cat))