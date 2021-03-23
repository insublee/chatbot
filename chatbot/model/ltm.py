import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LongTermMemory(torch.nn.Module):
    def __init__(self, LongTermMemory_size, sentence_emb_dim):
        super().__init__()
        self.sentence_emb_dim = sentence_emb_dim
        self.LongTermMemory_size = LongTermMemory_size
        self.temperature = 0.11 - torch.log10(torch.tensor(float(LongTermMemory_size))).item()*0.01 # 휴리스틱 하긴 함..
        self.age_noise = 4.0
        self.reward_scaling_factor = 10.0
        self.reward_age_base = 10.0
        self.age_discount = 1.0
        
        self.mem_build()
        

    def mem_build(self):
        self.keys = Parameter(F.normalize(torch.rand((self.LongTermMemory_size, self.sentence_emb_dim), 
                                           dtype=torch.float32)-0.5, dim=1), requires_grad=False)
        self.keys_var = Parameter(self.keys.data, requires_grad=False)
        self.values = Parameter(F.normalize(torch.rand((self.LongTermMemory_size, self.sentence_emb_dim), 
                                           dtype=torch.float32)-0.5, dim=1), requires_grad=False)
        self.values_var = Parameter(self.values.data, requires_grad=False)
        self.age = Parameter(torch.zeros((self.LongTermMemory_size), dtype=torch.float32),requires_grad=False)
        self.age += self.random_uniform((self.age.size()), -self.age_noise, self.age_noise)


    def forward(self,encoded_state):
        self.keys_var = Parameter(self.keys.data, requires_grad=False)
        self.values_var = Parameter(self.values.data, requires_grad=False)
        normalized_state = F.normalize(encoded_state,dim=1) # (batch, hidden)
        k_sim = torch.matmul(normalized_state, self.keys_var.T)# (batch, LongTermMemory_size)
        attention_weight = F.softmax(k_sim/self.temperature,dim=-1) # (batch, LongTermMemory_size)
        weighted_sum = torch.matmul(attention_weight, self.values_var) # (batch, hidden)
        self.age += torch.sum(attention_weight.detach(), dim=0) # (LongTermMemory_size)
        return weighted_sum

    def save_memory(self, encoded_state, encoded_action, reward):
        with torch.no_grad():
            batch_size = encoded_state.size(0)
            # oblivion
            self.age -= self.age_discount * batch_size * (1 / self.LongTermMemory_size) 
            # Get the least used memory index as much as the batch size.
            indexes = torch.topk(self.age, batch_size, largest=False)[1]
            #self.keys[indexes] = encoded_state # self.keys가 CPU에 있음.
            self.keys[indexes] = encoded_state
            #self.values[indexes] = encoded_action
            self.values[indexes] = encoded_action
            #print("torch.abs(reward).size() :", torch.abs(reward).size()) # torch.Size([4,1])
            self.age[indexes] = torch.abs(reward).squeeze(1) * self.reward_scaling_factor + self.reward_age_base

    
    def retrieve_from_logits(self,memory_logits):
        #return torch.matmul(memory_logits, self.values) # (batch, hidden)
        return torch.matmul(memory_logits, self.values_var) # (batch, hidden)

    
    def retrieve_index(self, encoded_action):
        encoded_action = F.normalize(encoded_action, dim=1)
        #return torch.argmax(torch.matmul(encoded_action, self.values.T), dim=1).unsqueeze(-1)
        return torch.argmax(torch.matmul(encoded_action, self.values_var.T), dim=1).unsqueeze(-1)

    def random_uniform(self, shape, low, high):
        return (high - low) * torch.rand(shape) + low