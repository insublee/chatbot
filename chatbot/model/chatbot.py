import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sacrebleu import corpus_bleu

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
        
        self.hparams = kwarg['hparams']
        self.hparams.sentence_emb_dim = self.model.config.d_model
        self.hparams.vocab_size = self.model.config.vocab_size
        self.hparams.padding_index = self.model.config.pad_token_id
        
        self.encoder = CustomEncoder(self.model.get_encoder())
        self.decoder = self.model.get_decoder()
        self.config = self.model.config
        #self.long_term_memory = LongTermMemory(kwarg.get('LongTermMemory_size'), self.config.hidden_size)
        
        self.long_term_memory = LongTermMemory(self.hparams)
        #self.selector = Selector(kwarg.get('LongTermMemory_size'), self.config.hidden_size)
        self.selector = Selector(self.hparams)
        self.sentiment_model = SentimentAstimater(self.hparams)
        self.sentiment_model.load_state_dict(torch.load('sentiment-analysis-en.pt'))
        
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
        memory_logits = self.selector(state_vector) # (batch, LTM_size) 
        # selected_action에 대한 학습은 강화학습 로스쪽에서 계산.
        selected_action = self.long_term_memory.retrieve_from_logits(memory_logits).unsqueeze(1) # (batch, 1, hidden)
        memory_mask = attention_mask[:,0].unsqueeze(1) # (batch, 1)

        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                           attention_mask=decoder_attention_mask,
                           encoder_hidden_states=selected_action,
                           encoder_attention_mask=memory_mask,
                           head_mask=head_mask,
                           encoder_head_mask=None,
                           past_key_values=past_key_values,
                           inputs_embeds=inputs_embeds,
                           use_cache=use_cache, 
                           output_attentions=None,
                           output_hidden_states=None,
                           return_dict=return_dict)
        
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        
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
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
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
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = ChatbotModel.from_pretrained("facebook/bart-base",**{"hparams":self.hparams})
    
    def forward(self, **input):
        return self.model(**input)
    
    def training_step(self, batch, batch_idx):
        state_lm_loss = self.language_model_loss(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        action_lm_loss = self.language_model_loss(input_ids=batch['decoder_input_ids'], attention_mask=batch['decoder_attention_mask'])
        policy_loss = self.policy_loss(**batch).mean()
        self.log('LM_loss', state_lm_loss + action_lm_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('RL_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True)

        return state_lm_loss + action_lm_loss + policy_loss

    def validation_step(self, batch, batch_idx):
        # . BLEU4 measures how many n-grams in a generated response overlap with those of the reference
        # 구해야 할것은 val_loss, BELU 스코어.
        # 이중에 val_loss는 인코더에만 인풋을 주고 디코더 아웃풋과 레이블을 비교한 loss
        model_output = self(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            decoder_input_ids=batch['decoder_input_ids'],
                            decoder_attention_mask=batch['decoder_attention_mask'],
                            labels=batch['labels'],
                            use_cache=False)
        # sys = self.model.generate(input_ids=batch['input_ids'])
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= self.hparams.eval_beams,
            max_length = self.hparams.max_seq_length,
            early_stopping = self.hparams.early_stopping,
            )
        sys = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ref = self.tokenizer.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        """
        sys : [" hi, how are you doing? i'm getting ready to do some cheetah chasing to stay in shape.", 
         ' i am! for my hobby i like to do canning or some whittling.',]
        ref : [' you must be very fast. hunting is one of my favorite hobbies.', 
         ' i also remodel homes when i am not out bow hunting.',]
        """
        bleu = corpus_bleu(sys, [ref])
        return {'loss': model_output['loss'], 'bleu':bleu.score}
    
    def language_model_loss(self, input_ids=None, attention_mask=None, **kwarg):
        loss_fct = CrossEntropyLoss()
        encoder_output = self.model.encoder(input_ids=input_ids,attention_mask=attention_mask)
        vector = self.model.encoder.vectorizing(encoder_output,attention_mask)
        mask = attention_mask[:,0].unsqueeze(1)
        decoder_outputs = self.model.decoder(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             encoder_hidden_states=vector,
                                             encoder_attention_mask=mask,
                                             )
        state_lm_logits = self.model.lm_head(decoder_outputs[0]) + self.model.final_logits_bias
        return loss_fct(state_lm_logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))
    
    
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
            action_encoder_output = self.model.encoder(input_ids=decoder_input_ids,
                                                attention_mask=decoder_attention_mask)
            encoded_action = self.model.encoder.vectorizing(action_encoder_output,
                                               decoder_attention_mask) # vectors
            action_idx = self.model.long_term_memory.retrieve_index(encoded_action) # 실제로 한 행동에대한 인덱스
            
            r = self.model.sentiment_model(next_state_input_ids) #리워드 추측
            # 저장
            self.model.long_term_memory.save_memory(encoded_action, r)

        
        # Vanilla Actor-Critic 
        td_target = r + self.hparams.gamma * self.model.v(input_ids=next_state_input_ids,
                                                  attention_mask=next_state_attention_mask)
        delta = td_target - self.model.v(input_ids=input_ids,
                                         attention_mask=attention_mask)

        pi = self.model.pi(input_ids=input_ids,attention_mask=attention_mask)
        pi_a = pi.gather(1,action_idx)
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