from tqdm.notebook import tqdm
import json
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_pretrained_bert import cached_path
from transformers import BartTokenizer
import datasets

"""
def CustomPersonachatDataset(sep_token):
    url="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
    personachat_file = cached_path(url)
    with open(personachat_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    for split in dataset.keys():
        state = []
        action = []
        next_state = []
        for i in tqdm(range(len(dataset[split]))): # train, val = 17000+1000
            for j in range(len(dataset[split][i]['utterances'])): # historys~20
                dialogue = dataset[split][i]['utterances'][j]['history']
                try :
                    dialogue.remove('__ SILENCE __')
                except ValueError:
                    pass
                dialogue_length = len(dialogue)
                if dialogue_length < 3:
                    continue
                state.append(dialogue[-3])
                action.append(dialogue[-2])
                next_state.append(dialogue[-1])

        if split == 'train':
            train_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
        elif split == 'valid':
            valid_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
    return datasets.DatasetDict({'train':train_dict, 'validation':valid_dict})
        
class DataModule(pl.LightningDataModule):
    1. prepare_data
      download tokenizer and persona-chat dataset
    2. setup
      tokenizeing & make batchs
    loader_columns = [
        'input_ids',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask',
        'labels',
        'next_state_input_ids',
        'next_state_attention_mask',
    ]

    def __init__(
        self,
        hparams,
        **kwargs
    ):
        super().__init__()
        self.hparams = hparams
    
    def prepare_data(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True, use_fast=True)
        self.dataset = CustomPersonachatDataset(self.tokenizer.sep_token)

    def setup(self, stage):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'valid' in x]

    def convert_to_features(self, example_batch, indices=None):
        # Tokenizing
        state_features = self.tokenizer.batch_encode_plus(
            example_batch['state'],
            max_length=self.hparams.max_seq_length,
            padding=True,
            truncation=True
        )
        
        action_features = self.tokenizer.batch_encode_plus(
            example_batch['action'],
            max_length=self.hparams.max_seq_length,
            padding=True,
            truncation=True
        )
        
        next_state_features = self.tokenizer.batch_encode_plus(
            example_batch['next_state'],
            max_length=self.hparams.max_seq_length,
            padding=True,
            truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features = {}
        features['input_ids'] = state_features['input_ids']
        features['attention_mask'] = state_features['attention_mask']
        features['decoder_input_ids'] = action_features['input_ids']
        features['decoder_attention_mask'] = action_features['attention_mask']
        features['labels'] = features['decoder_input_ids']
        features['next_state_input_ids'] = next_state_features['input_ids']
        features['next_state_attention_mask'] = next_state_features['attention_mask']

        return features

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.hparams.eval_batch_size)
"""
"""
def CustomPersonachatDataset(sep_token):
    url="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
    max_seq_length: int = 64
    max_dialogue_length: int = 3

    personachat_file = cached_path(url)
    with open(personachat_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    for split in dataset.keys():
        state = []
        action = []
        next_state = []
        for i in tqdm(range(len(dataset[split]))): # 이건 train, val 데이터 갯수. 17000+1000
            for j in range(len(dataset[split][i]['utterances'])): # 히스토리 적층구조로 대략 20개
                dialogue = dataset[split][i]['utterances'][j]['history']
                try :
                    dialogue.remove('__ SILENCE __')
                except ValueError:
                    pass
                dialogue_length = len(dialogue)
                if dialogue_length < 3:
                    continue
                state.append(dialogue[-3])
                action.append(dialogue[-2])
                next_state.append(dialogue[-1])
        # sentimental 모델이 쳇봇 바깥에서 reward를 계산한다면 이 주석 바로아래에서 next_state로 계산하면 됨

        if split == 'train':
            train_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
        elif split == 'valid':
            valid_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
    return datasets.DatasetDict({'train':train_dict, 'validation':valid_dict})
        
class DataModule(pl.LightningDataModule):
    1. prepare_data
      토크나이저랑 데이터셋 다운
    2. setup
      tokenizeing & train, validation, test셋 생성
    3. 트레인이나 발리데이션 데이터 로더로 불러서 씀.
    loader_columns = [
        'input_ids',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask',
        'labels',
        'next_state_input_ids',
        'next_state_attention_mask',
    ]

    def __init__(
        self,
        hparams,
        max_seq_length: int = 1024,
        max_dialogue_length: int = 9,

        **kwargs
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.random_seed=42
    
    def prepare_data(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True, use_fast=True)
        self.dataset = CustomPersonachatDataset(self.tokenizer.sep_token)

    def setup(self, stage):
        for split in self.dataset.keys():
            #print("self.dataset[split].column_names :", self.dataset[split].column_names)
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'valid' in x]

    def convert_to_features(self, example_batch, indices=None):
        # Tokenizing
        
        state_features = self.tokenizer.batch_encode_plus(
            example_batch['state'],
            max_length=self.max_seq_length,
            padding=True,
            #pad_to_max_length=True,
            truncation=True
        )
        
        action_features = self.tokenizer.batch_encode_plus(
            example_batch['action'],
            max_length=self.max_seq_length,
            padding=True,
            #pad_to_max_length=True,
            truncation=True
        )
        
        next_state_features = self.tokenizer.batch_encode_plus(
            example_batch['next_state'],
            max_length=self.max_seq_length,
            padding=True,
            #pad_to_max_length=True,
            truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features = {}
        features['input_ids'] = state_features['input_ids']
        features['attention_mask'] = state_features['attention_mask']
        features['decoder_input_ids'] = action_features['input_ids']
        features['decoder_attention_mask'] = action_features['attention_mask']
        features['labels'] = features['decoder_input_ids']
        features['next_state_input_ids'] = next_state_features['input_ids']
        features['next_state_attention_mask'] = next_state_features['attention_mask']

        return features

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)
"""

def CustomPersonachatDataset(sep_token):
    url="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

    personachat_file = cached_path(url)
    with open(personachat_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    for split in dataset.keys():
        state = []
        action = []
        next_state = []
        for i in tqdm(range(len(dataset[split]))): # 이건 train, val 데이터 갯수. 17000+1000
            for j in range(len(dataset[split][i]['utterances'])): # 히스토리 적층구조로 대략 20개
                dialogue = dataset[split][i]['utterances'][j]['history']
                try :
                    dialogue.remove('__ SILENCE __')
                except ValueError:
                    pass
                dialogue_length = len(dialogue)
                if dialogue_length < 3:
                    continue
                state.append(dialogue[-3])
                action.append(dialogue[-2])
                next_state.append(dialogue[-1])
        # sentimental 모델이 쳇봇 바깥에서 reward를 계산한다면 이 주석 바로아래에서 next_state로 계산하면 됨

        if split == 'train':
            train_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
        elif split == 'valid':
            valid_dict = datasets.Dataset.from_dict({'state' : state, 'action' : action, 'next_state' : next_state})
    return datasets.DatasetDict({'train':train_dict, 'validation':valid_dict})
        
class DataModule(pl.LightningDataModule):
    """
    1. prepare_data
      토크나이저랑 데이터셋 다운
    2. setup
      tokenizeing & train, validation, test셋 생성
    3. 트레인이나 발리데이션 데이터 로더로 불러서 씀.
    """
    loader_columns = [
        'input_ids',
        'attention_mask',
        'decoder_input_ids',
        'decoder_attention_mask',
        'labels',
        'next_state_input_ids',
        'next_state_attention_mask',
    ]

    def __init__(
        self,
        hparams,
        max_seq_length: int = 64,
        max_dialogue_length: int = 9,
        **kwargs
    ):
        super().__init__()
        self.hparams = hparams
        self.random_seed=42
    
    def prepare_data(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True, use_fast=True)
        self.dataset = CustomPersonachatDataset(self.tokenizer.sep_token)

    def setup(self, stage):
        for split in self.dataset.keys():
            #print("self.dataset[split].column_names :", self.dataset[split].column_names)
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'valid' in x]

    def convert_to_features(self, example_batch, indices=None):
        # Tokenizing
        
        state_features = self.tokenizer.batch_encode_plus(
            example_batch['state'],
            max_length=self.hparams.max_seq_length,
            #padding=True,
            pad_to_max_length=True,
            truncation=True
        )
        
        action_features = self.tokenizer.batch_encode_plus(
            example_batch['action'],
            max_length=self.hparams.max_seq_length,
            #padding=True,
            pad_to_max_length=True,
            truncation=True
        )
        
        next_state_features = self.tokenizer.batch_encode_plus(
            example_batch['next_state'],
            max_length=self.hparams.max_seq_length,
            #padding=True,
            pad_to_max_length=True,
            truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features = {}
        features['input_ids'] = state_features['input_ids']
        features['attention_mask'] = state_features['attention_mask']
        features['decoder_input_ids'] = action_features['input_ids']
        features['decoder_attention_mask'] = action_features['attention_mask']
        features['labels'] = features['decoder_input_ids']
        features['next_state_input_ids'] = next_state_features['input_ids']
        features['next_state_attention_mask'] = next_state_features['attention_mask']

        return features

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.hparams.eval_batch_size)