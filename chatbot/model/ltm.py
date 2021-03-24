import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LongTermMemory(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.temperature = 0.11 - torch.log10(torch.tensor(float(self.hparams.LongTermMemory_size))).item()*0.01 # 휴리스틱 하긴 함..
        #self.age_noise = 4.0
        #self.reward_scaling_factor = 10.0
        #self.reward_age_base = 10.0
        #self.age_discount = 1.0
        #self.hit_th = 0.9
        
        self.mem_build()
        

    def mem_build(self):
        self.values = Parameter(F.normalize(torch.rand((self.hparams.LongTermMemory_size, self.hparams.sentence_emb_dim), 
                                           dtype=torch.float32)-0.5, dim=1), requires_grad=False)
        self.values_var = Parameter(self.values.data, requires_grad=False)
        self.age = Parameter(torch.zeros((self.hparams.LongTermMemory_size), dtype=torch.float32),requires_grad=False)
        self.age += self.random_uniform((self.age.size()), -self.hparams.age_noise, self.hparams.age_noise)

    def forward(self,encoded_action):
        normalized_action = F.normalize(encoded_action,dim=1) # (batch, hidden)
        sim = torch.matmul(normalized_action, self.values_var.T)# (batch, LongTermMemory_size)
        attention_weight = F.softmax(sim/self.temperature,dim=-1) # (batch, LongTermMemory_size)
        weighted_sum = torch.matmul(attention_weight, self.values_var) # (batch, hidden)
        self.age += torch.sum(attention_weight.detach(), dim=0) # (LongTermMemory_size)
        return weighted_sum

    def save_memory(self, encoded_action, reward):
        with torch.no_grad():
            batch_size = encoded_action.size(0)
            # oblivion
            self.age -= self.hparams.age_discount * batch_size * (1 / self.hparams.LongTermMemory_size) 
            # encoded_action이 기존의 인덱스와 비슷하다면 새로저장 X
            normalized_action = F.normalize(encoded_action,dim=1) # (batch, hidden)
            sim = torch.matmul(normalized_action, self.values_var.T)# (batch, LongTermMemory_size)
            idxs = torch.matmul(sim.ge(self.hparams.hit_th).float(), torch.ones((sim.size(1),1), device=sim.device))
            hit_msk = torch.clamp(idxs, min=0, max=1)
            hit_index = hit_msk.nonzero()[:,0]
            miss_msk = 1 - hit_msk
            miss_index = miss_msk.nonzero()[:,0]

            # Get the least used memory index as much as the batch size.
            useless_memory_indexes = torch.topk(self.age, len(miss_index), largest=False)[1]
            self.values[useless_memory_indexes] = normalized_action[miss_index]
            self.age[useless_memory_indexes] = torch.abs(reward[miss_index]).squeeze(1) * self.hparams.reward_scaling_factor + self.hparams.reward_age_base
    
    def retrieve_from_logits(self,memory_logits):
        return torch.matmul(memory_logits, self.values_var) # (batch, hidden)
    
    def retrieve_index(self, encoded_action):
        encoded_action = F.normalize(encoded_action, dim=1)
        #return torch.argmax(torch.matmul(encoded_action, self.values.T), dim=1).unsqueeze(-1)
        return torch.argmax(torch.matmul(encoded_action, self.values_var.T), dim=1).unsqueeze(-1)

    def random_uniform(self, shape, low, high):
        return (high - low) * torch.rand(shape) + low