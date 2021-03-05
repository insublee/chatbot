import torch
import torch.nn.functional as F


class LongTermMemory(torch.nn.Module):
    """
    idea from LSH_Memory and NTM
    LTM = LongTermMemory(7, 10, 768, 64)
    test_state = torch.rand((4,768))
    test_action = torch.rand((4,768))
    test_reward = torch.rand((4))*1000
    print(LTM.age)
    print(LTM(test_state).size())
    LTM.save_memory(test_state, test_action, test_reward)
    print(LTM.age)
    """
    def __init__(
        self,
        WM_size,
        LongTermMemory_size,
        sentence_emb_dim,
        max_seq_length,
        ):
        super().__init__()
        self.sentence_emb_dim = sentence_emb_dim
        self.LongTermMemory_size = LongTermMemory_size
        self.temperature = 0.11 - torch.log10(torch.tensor(float(LongTermMemory_size))).item()*0.01 # 휴리스틱 하긴 함..
        self.age_noise = 4.0
        self.reward_scaling_factor = 10.0
        self.reward_age_base = 10.0
        # aging은 메모리 전체 합해서 1이므로 깎여나가는 aging은 메모리 크기에 반비례 해야함
        self.age_discount = 1.0
        self.mem_build()
        

    def mem_build(self):
        self.keys = F.normalize(torch.rand((self.LongTermMemory_size, self.sentence_emb_dim), requires_grad=False, dtype=torch.float32)-0.5, dim=1)
        self.values = F.normalize(torch.rand((self.LongTermMemory_size, self.sentence_emb_dim), requires_grad=False, dtype=torch.float32)-0.5, dim=1)
        self.age = torch.zeros((self.LongTermMemory_size), requires_grad=False, dtype=torch.float32)
        self.age += self.random_uniform((self.age.size()), -self.age_noise, self.age_noise)
    
    """
    [state0                 |action0]       [2.1]
    [안녕하세요             |인사 잘하네~]  [9.2]
    [내 사과 화장실에 있어  |응 알았어]     [28.3] 
    
    감정에 연관된 기억이거나, 새로운기억이거나, 사용이 많이 된 기억들은 age값이 높다.
    age는 전체적으로 감소한다.
    """


    def forward(self,encoded_state):
        # normalize encoded_state
        normalized_state = F.normalize(encoded_state,dim=1) # (batch, hidden)
        # weighted sum
        k_sim = torch.matmul(normalized_state, self.keys.T)# (batch, LongTermMemory_size)
        attention_weight = F.softmax(k_sim/self.temperature,dim=-1) # (batch, LongTermMemory_size)
        weighted_sum = torch.matmul(attention_weight, self.values) # (batch, hidden)
        # 뉴럴튜링머신처럼 어텐션분포로 메모리 기억을 업데이트함. 1*batch 만큼 aging됨.
        # Aging used memory (NTM style)
        self.age += torch.sum(attention_weight, dim=0) # (LongTermMemory_size)
        return weighted_sum

    def save_memory(self, encoded_state, encoded_action, reward):
        """
        reward 의 절대값에 비례해서 age를 준다.(감정이 연관된 기억은 오래 저장.)
        메모리의 새로 저장할 인덱스는 가장 사용 하지 않는 (age가 가장 작은것)메모리를 사용.
        자주 사용된 메모리는 불러올때마다 에이지가 늘어나기 때문에 값이 크다.
        전체적인 메모리 age는 시간이 지날때마다 감소시키면 쓸모없는 기억은 날아간다.
        전체적인 메모리 age는 forward 시 어텐션분포 * 배치사이즈만큼 더해지므로 감소는 그만큼 해줘야한다.
        감정이랑 연관되지 않는 기억이 들어오더라도, 새로운 기억은 어느정도 가지고 있어야 하기 때문에 reward_age_base로 초기 값을 저장해준다.
        """
        batch_size = encoded_state.size(0)
        # oblivion
        self.age -= self.age_discount * batch_size * (1 / self.LongTermMemory_size) 
        # Get the least used memory index as much as the batch size.
        indexes = torch.topk(self.age, batch_size, largest=False)[1]
        # Assign actions and states to the corresponding index.
        self.keys[indexes] = encoded_state
        self.values[indexes] = encoded_action
        #print("torch.abs(reward).size() :", torch.abs(reward).size()) # torch.Size([4,1])
        self.age[indexes] = torch.abs(reward).squeeze(1) * self.reward_scaling_factor + self.reward_age_base

    
    def retrieve_from_logits(self,memory_logits):
        return torch.matmul(memory_logits, self.values) # (batch, hidden)
    
    def retrieve_index(self, encoded_action):
        # 실제로 선택했던 action에 대한 index를 구해야 하므로 메모리 value값을 통해서 봄.
        # 실제대화 state : 어른을 보면 인사를 해야지!, action : 싫어
        # (key0 : 어른을 보면 인사를 해야지!, value0 : 싫은데요)
        # (key1 : 어른을 보면 인사를 해야지!, value1 : 죄송합니다. 다음부터 인사 잘할게요)
        # retrieved index = 0
        encoded_action = F.normalize(encoded_action, dim=1)

        return torch.argmax(torch.matmul(encoded_action, self.values.T), dim=1)

    def random_uniform(self, shape, low, high):
        return (high - low) * torch.rand(shape) + low