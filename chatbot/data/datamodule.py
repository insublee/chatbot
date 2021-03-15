from argparse import ArgumentParser
from datetime import datetime
from typing import Optional
from tqdm.notebook import tqdm
from pprint import pprint
import json

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import BartTokenizer
from pytorch_pretrained_bert import cached_path
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

class CustomPersonachatDataset(Dataset): 
    def __init__(self, encoder_model_name_or_path, use):

        self.url="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

        self.max_seq_length: int = 64
        self.max_dialogue_length: int = 9
        self.use = use
        
        #self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name_or_path, use_fast=True)        
        self.encoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True, use_fast=True)
        self.prepare_data()

    def prepare_data(self):
        # 단순히 다이얼로그만 제공
        personachat_file = cached_path(self.url)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        self.chat_data=[]

        for split in dataset.keys():
            for i in tqdm(range(int(len(dataset[split])*self.use))): # 이건 train, val 데이터 갯수. 17000+1000
                for j in range(len(dataset[split][i]['utterances'])): # 히스토리 적층구조로 대략 20개
                    # 다이얼로그 히스토리 [str,str,...]
                    dialogue = dataset[split][i]['utterances'][j]['history']
                    # 길면 잘라줌. 토크나이저에 넣을거라서 패딩은 아직 하지 않음. 히스토리가 3개 미만이면 too_short==True
                    try :
                        dialogue.remove('__ SILENCE __')
                    except ValueError:
                        pass
                    dialogue, dialogue_mask, too_short = self.cutter(dialogue) 
                    if not too_short:
                        #WM, state, action, next_state = dialogue[:7], dialogue[6], dialogue[7], dialogue[8]
                        #print(f"WM:\t\t{WM}\nSTATE:   \t{state}\nACTION:   \t{action}\nNEXT_STATE:\t{next_state}\n")
                        # 다이얼로그를 토크나이징 함. features의 각 길이는 최대 9개 최소 3개.
                        features = self.encoder_tokenizer.batch_encode_plus(
                            dialogue,
                            max_length=self.max_seq_length,
                            pad_to_max_length=True,
                            truncation=True,
                            )
                        # 다이얼로그 마스크 추가. 다이얼로그 마스크의 길이는 9개.
                        features['dialogue_mask'] = dialogue_mask
                        
                        # 강화학습시 사용할 감성분석데이터. features에 넣어줌. 
                        #features['sentiments'] = self.sentiment_analysis(dialogue)

                        # 다이얼로그 패딩
                        features = self.padding(features)

                        # 결국 데이터포인트 하나는 {input_ids, token_type_ids, attention_mask, sentiments, dialogue_mask}로 이루어져있음.
                        self.chat_data.append(features)
                    else:
                        pass
        #torch.save(self.chat_data, 'chat_data.pt')
        self.dataset_size = len(self.chat_data)
        

    def sentiment_analysis(self, dialogue):
        # state에 해당하는 dialogue[-3]
        state = [dialogue[-3]]
        sentiment_features = self.sentiment_tokenizer.batch_encode_plus(
            state,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            #return_tensors='pt'
            )
        output = self.sentiment_model(**sentiment_features)
        logits = F.softmax(output.logits, dim=1)
        sentiment = torch.matmul(logits, torch.arange(1,6).unsqueeze(dim=1).float()) - 3 # normalize
        return sentiment.squeeze(dim=1).tolist()

    def cutter(self, dialogue):
        # cutoff dialogue
        # W.M.(0:7)+action(7:8)+next_state(8:9) max dialogue length = 7+1+1 = 9
        # 너무짧으면 셈플 무시할수 있도록 too_short=True 해줌
        dialogue_length = len(dialogue)
        too_short = False
        dialogue_mask = []
        if dialogue_length < 3:
            too_short = True
            return dialogue, dialogue_mask, too_short
        if dialogue_length < self.max_dialogue_length:
            padding_len = self.max_dialogue_length - dialogue_length
            dialogue_mask = [0]*padding_len + [1]*dialogue_length
        elif dialogue_length > self.max_dialogue_length:
            dialogue = dialogue[-self.max_dialogue_length:]
            dialogue_mask = [1]*(self.max_dialogue_length)
        else:
            dialogue_mask = [1]*(self.max_dialogue_length)
        
        return dialogue, dialogue_mask, too_short


    def padding(self, features):
        dialogue_length = len(features['input_ids'])
        # 다이얼로그가 짧을경우 앞쪽으로 패딩
        if dialogue_length < self.max_dialogue_length:
            padding_len = self.max_dialogue_length - dialogue_length
            for k in features.keys():
                if k=='dialogue_mask':
                    pass
                else:
                    features[k] = [[0] * self.max_seq_length] * padding_len + features[k]
                    #features[k] = torch.tensor([[0] * self.max_seq_length] * padding_len + features[k], dtype=torch.long)
                    
        return features
        
    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.chat_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴. 딱 하나만 가져오는거임
    def __getitem__(self, idx):
        sample={}
        for key in self.chat_data[idx].keys():
            #sample[key] = torch.FloatTensor(self.chat_data[idx][key])
            sample[key] = torch.LongTensor(self.chat_data[idx][key])
        return sample

class DataModule(pl.LightningDataModule):
    """
    1. prepare_data
      토크나이저랑 데이터셋 다운 이때 데이터는 토크나이징 되어있음.
    2. setup
      prepare_data 에서 받은 데이터로 train, validation, test셋 생성
    3. 트레인이나 발리데이션 데이터 로더로 불러서 씀.
    """


    def __init__(
        self,
        model_name_or_path : str,
        max_seq_length: int = 64,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        test_batch_size: int = 4,
        validation_split:float=0.2,
        test_split:float=0.2,
        shuffle_dataset=True,
        use:float=1,
        **kwargs
    ):
        super().__init__()
        self.use = use
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.validation_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed=42
        self.shuffle_dataset=True

        #self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name_or_path, use_fast=True)
        #self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name_or_path, use_fast=True)

    def setup(self, stage):
        dataset_size = self.dataset.dataset_size
        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices = list(range(dataset_size))
        val_split = int(np.floor(self.validation_split * dataset_size))
        test_split = int(np.floor(self.test_split * dataset_size))

        if self.shuffle_dataset :
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            #[8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split], 
        test_indices = indices[val_split:val_split + test_split]
        #[8, 1] [5, 0] [7, 2, 9, 4, 3, 6] 
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def prepare_data(self):
        self.dataset = CustomPersonachatDataset(
            encoder_model_name_or_path=self.model_name_or_path, use = self.use)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.train_batch_size,
                          sampler=self.train_sampler,
                         num_workers=40)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.val_batch_size, 
                          sampler=self.valid_sampler,
                         num_workers=40)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.test_batch_size, 
                          sampler=self.test_sampler,
                         num_workers=40)