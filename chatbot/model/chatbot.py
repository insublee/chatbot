import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import pytorch_lightning as pl
import datasets
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput

from .encoder import CustomEncoder
from .ltm import LongTermMemory
from .selector import Selector
from .sentiment import SentimentAstimater


class ChatbotModel(BartForConditionalGeneration):
    def __init__(self, config, **kwarg):
        super().__init__(config)

        self.encoder = CustomEncoder(self.model.get_encoder())
        self.decoder = self.model.get_decoder()
        self.config = self.model.config
        self.long_term_memory = LongTermMemory(kwarg.get('LongTermMemory_size'),self.config.hidden_size)
        self.selector = Selector(kwarg.get('LongTermMemory_size'), self.config.hidden_size)
        self.sentiment_model = SentimentAstimater()
        
        #del self.model
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def pi(self, input_ids=None, attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        x = self.encoder.vectorizing(encoder_outputs,attention_mask) # vectors
        x = self.long_term_memory(x)
        x = F.relu(self.selector.fc_pi_1(x))
        x = self.selector.fc_pi_2(x)
        prob = F.softmax(x, dim=1)
        return prob

    def v(self, input_ids=None, attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        x = self.encoder.vectorizing(encoder_outputs,attention_mask) # vectors
        x = F.relu(self.selector.fc_v_1(x))
        v = self.selector.fc_v_2(x)
        return v

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwarg
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        action_vector, reward = None, None
        # 저장은 decoder_input_ids 있고 encoder_outputs는 없고, 레이블은 있을때 한다.
        if decoder_input_ids is not None and encoder_outputs is None and labels is not None and kwarg.get('next_state_input_ids') is not None:
            with torch.no_grad():
                encoded_action = self.encoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    )
                action_vector = self.encoder.vectorizing(encoded_action, decoder_attention_mask) # (batch,hidden)
                
                encoded_next_state = self.encoder(
                    input_ids=kwarg['next_state_input_ids'],
                    attention_mask=kwarg['next_state_attention_mask'],
                    )
                reward = self.sentiment_model(encoded_next_state[0])


        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if encoder_outputs is None: # 빔서치할때 이미 인코더로 encoder_outputs 만들어줌
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
        
        
        state_vector = self.encoder.vectorizing(encoder_outputs, attention_mask) # (batch,hidden)
        retrieved_action = self.long_term_memory(state_vector) # (batch,hidden)
        # 디코더가 말을 만들어낼 때는 (encoder_outputs + selected_action)를 보는것이 아니라
        # (encoder_outputs + retrieved_action)로 만들어야함.
        # 인코더 학습과정은 retrieved_action, 즉 저장된 정보에서 말을 만들어내야 하기 때문.
        # selected_action에 대한 학습은 강화학습 로스쪽에서 계산.
        
        concated_hidden_states = torch.cat((encoder_outputs[0], retrieved_action.unsqueeze(1)), dim=1) # (batch,seq+1,hidden)
        selected_mask = attention_mask[:,0].unsqueeze(1) # 메모리 덧붙인거 1로 해줘야하는데 디바이스 문제로 이렇게 해줌.
        # selected_mask는 (batch, 1) 이고 1로만 있어야함       # selected_mask는 (batch, 1) 이고 1로만 있어야함
        concated_state_mask = torch.cat((attention_mask,selected_mask),dim=1)

        decoder_outputs = self.decoder(input_ids=decoder_input_ids, # 디코더 인풋아이디
                           attention_mask=decoder_attention_mask, # 디코더 인풋아이디랑 같이들어갈 어텐션 마스크
                           encoder_hidden_states=concated_hidden_states, # 벡터라이징 된 WM과 selected 된 action vector
                           encoder_attention_mask=concated_state_mask, # 위어꺼 마스크 꼭줘야함
                           head_mask=head_mask, # 멀티헤드 어텐션 모듈 마스큰데 이거 필요없음.
                           encoder_head_mask=None, # 크로스어텐션 관련 마스큰데 이거도 필요없음
                           past_key_values=past_key_values, # 디코딩할때 속도내려고 하는건데 필요없음
                           inputs_embeds=inputs_embeds, # 인풋아이디 안주고 이거로 넘길수 있는데 필요없음
                           use_cache=use_cache, 
                           output_attentions=None, # 모든 레이어에 대한 어텐션인데, 필요없음
                           output_hidden_states=None, # 전체적인 레이어 히든스테이트 값들인데, 필요없음
                           return_dict=return_dict)
        
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        if decoder_input_ids is not None and labels is not None and kwarg.get('next_state_input_ids') is not None:
            self.long_term_memory.save_memory(state_vector, action_vector, reward)

        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            #encoder_hidden_states=encoder_outputs.hidden_states,
            #encoder_attentions=encoder_outputs.attentions,
        )
    def shift_tokens_right(self,input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    
class Chatbot(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        kwarg = {"LongTermMemory_size":self.hparams.LongTermMemory_size}
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = ChatbotModel.from_pretrained("facebook/bart-base",**kwarg)
        #self.f1_metric = datasets.load_metric("f1")
    
    def forward(self, **input):
        return self.model(**input)
    
    def training_step(self, batch, batch_idx):
        LM_loss = self(**batch)[0]
        policy_loss = self.policy_loss(**batch)
        return LM_loss + policy_loss.mean()

    def validation_step(self, batch, batch_idx):
        # . BLEU4 measures how many n-grams in a generated response overlap with those of the reference
        # 구해야 할것은 val_loss, BELU 스코어.
        # 이중에 val_loss는 인코더에만 인풋을 주고 디코더 아웃풋과 레이블을 비교한 loss
        model_output = self(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            use_cache=False)
        # 이건 제일 마지막에 
        # sys = self.model.generate(input_ids=batch['input_ids'])
        # bleu = corpus_bleu(sys, refs)

        return {'loss': model_output['loss']}
    
    def policy_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    decoder_input_ids=None,
                    decoder_attention_mask=None,
                    next_state_input_ids=None,
                    next_state_attention_mask=None,
                    labels=None,
                    **kwarg):

        with torch.no_grad():
            encoder_output = self.model.encoder(input_ids=decoder_input_ids,
                                                attention_mask=decoder_attention_mask)
            a = self.model.encoder.vectorizing(encoder_output,
                                               decoder_attention_mask) # vectors
            a = self.model.long_term_memory.retrieve_index(a) # 실제로 한 행동에대한 인덱스
            
            r = self.model.sentiment_model(encoder_output[0]) # 여기 원래 인풋은 next_state_input_ids. 지금은 그냥 넣자
        
        # Vanilla Actor-Critic 
        td_target = r + self.hparams.gamma * self.model.v(input_ids=next_state_input_ids,
                                                  attention_mask=next_state_attention_mask)
        delta = td_target - self.model.v(input_ids=input_ids,
                                         attention_mask=attention_mask)

        pi = self.model.pi(input_ids=input_ids,
                           attention_mask=attention_mask)
        pi_a = pi.gather(1,a)
        pg = -torch.log(pi_a) * delta.detach()
        v_loss = F.smooth_l1_loss(self.model.v(input_ids=input_ids,attention_mask=attention_mask) , td_target.detach())

        return pg + v_loss
    
    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )
    
    def configure_optimizers(self):
        # https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb#scrollTo=gtn5YGKYO65B
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]