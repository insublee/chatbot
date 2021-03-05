import torch
import torch.nn.functional as F

class Selector(torch.nn.Module):
    def __init__(
        self,
        WM_size, 
        LongTermMemory_size, 
        sentence_emb_dim, 
        max_seq_length,
        ):
        super().__init__()
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps_clip = 0.1
        self.WM_size = WM_size
        self.sentence_emb_dim = sentence_emb_dim
        #print("selector : sentence_emb_dim :" , sentence_emb_dim)
        self.transformer_block = torch.nn.TransformerEncoderLayer(d_model=sentence_emb_dim, nhead=8)
        self.fc_pi = torch.nn.Linear((self.WM_size+1)*self.sentence_emb_dim, LongTermMemory_size)
        self.fc_1 = torch.nn.Linear(self.sentence_emb_dim, self.sentence_emb_dim * 4)
        self.fc_v = torch.nn.Linear(self.sentence_emb_dim * 4,1)

    def forward(self, encoded_WM, retrieved_action):
        # input : encoded_WM (batch, W.M.length, sentence_emb_dim)
        #         retrieved_action (batch, sentence_emb_dim)
        # output : memory_logits (batch, LongTermMemory_size)
        #print("selector : encoded_WM.size(), retrieved_action.size() :", encoded_WM.size(), retrieved_action.size())
        concated_feature = torch.cat((encoded_WM, retrieved_action.unsqueeze(1)), 1) # (batch, W.M.length + 1, sentence_emb_dim)
        #print("selector::concated_feature.transpose(1,0) : ", concated_feature.transpose(1,0).size(), concated_feature[0])
        #print("selector::concated_feature.norm : ", torch.norm(concated_feature,dim=1))
        #transformer_block 지금 nan이 속출하고있습니다! 왜그런지 알아봐야함. 레이어놈 안해서 그런가?
        out = self.transformer_block(concated_feature.transpose(1,0)) # (batch, W.M.length + 1, sentence_emb_dim)
        #print("selector:: out.size() : ", out.size(), out[0])
        
        out = out.reshape(concated_feature.size(0), concated_feature.size(1) * self.sentence_emb_dim)#(batch, W.M.length + 1* sentence_emb_dim)
        
        out = F.relu(self.fc_pi(out))#(batch, LongTermMemory_size)
        #print("selector : out.size() : ", out.size())
        memory_logits = F.softmax(out, dim=1)
        #print("selector:: memory_logits.size() : ", memory_logits.size(), memory_logits[0])
        return memory_logits

    def v(self, x):
        x = F.relu(self.fc_1(x))
        v = self.fc_v(x)
        return v
    
    def policy_loss(self, encoded_WM, retrieved_action, encoded_state, label_index, reward, encoded_next_state, selected_memory_logits):
        # https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
        # encoded_state : torch.Size([batch, sentence_emb_dim])
        # label_index : torch.Size([batch]) 
        # reward : torch.Size([batch]) 
        # encoded_next_state : torch.Size([batch, sentence_emb_dim])
        # memory_logits : torch.Size([batch, 1000]) 
        td_target = reward + self.gamma * self.v(encoded_next_state)
        delta = td_target - self.v(encoded_state)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        
        pi = self.forward(encoded_WM, retrieved_action)
        pi_a = pi.gather(1, label_index.unsqueeze(-1))
        ratio = torch.exp(torch.log(pi_a) - torch.log(selected_memory_logits))  # a/b == exp(log(a)-log(b))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
        
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(encoded_state) , td_target.detach())
        
        return loss.mean()