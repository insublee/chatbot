import torch.nn as nn
import torch

#import spacy

#from config import MIN_LENGTH
#from data import load_dataset

class SentimentAstimater(nn.Module):
    def __init__(
        self,
        vocab_size,
        sentence_emb_dim, 
        unknown_index,
        padding_index,
        ):
        super().__init__()

        self.model = SentimentAnalysisCNN(vocab_size, sentence_emb_dim, unknown_index, padding_index)
        self.model.load_state_dict(torch.load('sentiment-analysis-en.pt'))

    def forward(self,x):
        model_out = self.model(x)
        prediction = torch.sigmoid(model_out)
        result = prediction.item() * 2.0 - 1.0
        return result

"""


nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, TEXT, sentence):
    model.eval()

    # Tokenize
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    # Add padding
    if len(tokenized) < MIN_LENGTH:
        tokenized += ['<pad>'] * (MIN_LENGTH - len(tokenized))

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).unsqueeze(0)

    prediction = torch.sigmoid(model(tensor))
    result = prediction.item() * 2.0 - 1.0

    return result

def main():
    dataset, TEXT, LABEL = load_dataset()

    INPUT_DIM = len(TEXT.vocab)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = SentimentAnalysisCNN(INPUT_DIM, UNK_IDX, PAD_IDX)

    model.load_state_dict(torch.load('sentiment-analysis-en.pt'))
    
    # 여기서 문장 하나씩 집어넣으면 됨
    test_sentence = 'You are a bad guy.'

    predict_result = predict_sentiment(
        model = model,
        TEXT = TEXT,
        sentence = test_sentence,
        )

    print(f'\nTest sentence : {test_sentence}')
    print(f'Predict result : {predict_result:.5f}')


if __name__ == "__main__":
    print('\n-- START Testing Main --\n')
    main()
    print('\n-- FINISH Testing Main --\n\n')
"""
    
class SentimentAnalysisCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        sentence_emb_dim,
        unknown_index,
        padding_index,
        ):
        self.N_FILTERS = 100
        self.FILTER_SIZES = [3, 5, 7]
        self.OUTPUT_DIM = 1
        self.DROPOUT = 0.5
        
        super().__init__()
                
        self.embedding = nn.Embedding(
            vocab_size,
            sentence_emb_dim,
            padding_idx = padding_index,
            )

        self.embedding.weight.data[unknown_index] = torch.zeros(sentence_emb_dim)
        self.embedding.weight.data[padding_index] = torch.zeros(sentence_emb_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1, 
                out_channels = self.N_FILTERS, 
                kernel_size = (fs, sentence_emb_dim),
                ) 
            for fs in self.FILTER_SIZES
            ]) # 아 list comprihention이였구나
        
        self.dropout = nn.Dropout(self.DROPOUT)
        
        self.fc = nn.Linear(len(self.FILTER_SIZES) * self.N_FILTERS, self.OUTPUT_DIM)
        
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
        return self.fc(cat)