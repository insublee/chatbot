import pytorch_lightning as pl
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig, get_linear_schedule_with_warmup

from .encoder import CustomEncoder
from .ltm import LongTermMemory
from .selector import Selector
from .decoder import CustomDecoder
from .sentiment import SentimentAstimater



class Chatbot(pl.LightningModule):

    def __init__(
        self,
        model_name :str =  "facebook/bart-base",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        max_epochs: int = 3,
        gpus: int = 4,
        train_batch_size: int = 32,
        accumulate_grad_batches=1,
        
        WM_size :int = 7,
        LongTermMemory_size :int = 1000,
        sentence_emb_dim : int = 768,
        max_seq_length :int = 64,
        **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.long_term_memory = LongTermMemory(WM_size, 
                                               LongTermMemory_size, 
                                               sentence_emb_dim, 
                                               max_seq_length)
        self.selector = Selector(WM_size, 
                                 LongTermMemory_size, 
                                 sentence_emb_dim, 
                                 max_seq_length)
        
        # https://huggingface.co/facebook/bart-base/blob/main/config.json
        model = BartForConditionalGeneration.from_pretrained(model_name)
        encoder=model.get_encoder()
        decoder=model.get_decoder()
        self.config = model.config
        self.sentence_emb_dim = self.config.d_model
        self.encoder = CustomEncoder(encoder, 
                                     WM_size, 
                                     LongTermMemory_size, 
                                     sentence_emb_dim, 
                                     max_seq_length)
        self.decoder = CustomDecoder(decoder, 
                                     WM_size, 
                                     LongTermMemory_size, 
                                     sentence_emb_dim, 
                                     max_seq_length)
        self.lm_head = torch.nn.Linear(self.sentence_emb_dim, self.config.vocab_size, bias=False) 
        self.loss_function = torch.nn.CrossEntropyLoss()
        tokenizer = BartTokenizer.from_pretrained(model_name)
        self.sentiment_model = SentimentAstimater(self.config.vocab_size,
                                                  self.sentence_emb_dim, 
                                                  tokenizer.unk_token_id,
                                                  self.config.pad_token_id)

        # encoder -> LTM -> selector -> decoder(batch, seq, hidden) -> lm_head(batch, seq, vocab_size) -> seq generate
        
    def encoder_decoder_loss(self, decoder_last_hidden_state, action_ids):
        # decoder_last_hidden_state : (batch, seq, hidden)
        # action_ids : (batch, seq)
        lm_out = self.lm_head(decoder_last_hidden_state)
        return self.loss_function(lm_out.view(-1, self.config.vocab_size),action_ids.reshape(action_ids.size(0) * action_ids.size(1)))
        
        


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions

        # x = {input_ids(batch, max_dialogue_length, max_seq_length),
        #      token_type_ids(batch, max_dialogue_length, max_seq_length),
        #      attention_mask(batch, max_dialogue_length, max_seq_length),
        #      dialogue_mask(batch, max_dialogue_length)}
        
        # sentence encode
        x = self.encoder(x) # (batch, max_dialogue_length, sentence_emb_dim)
        batch_size = len(x)
        assert x.size()==torch.Size([batch_size, self.hparams.WM_size+2, self.sentence_emb_dim])

        # readjustment encoded batch
        encoded_WM, encoded_state, encoded_action, encoded_next_state = self.feature_preprocess(x) 

        # extract reward from small pretrained sentiment analisis model ex) thank you! +1, i don't understand.. -.5
        reward = self.sentiment_model(encoded_next_state).detach() # (batch)
        assert reward.size() == torch.Size([batch_size, 1])

        # save the memory by considering the reward.
        self.long_term_memory.save_memory(encoded_state, encoded_action, reward)

        # retrieve memory and select similar one
        retrieved_action = self.long_term_memory(encoded_state) # (batch,hidden)
        assert retrieved_action.size() == torch.Size([batch_size, self.sentence_emb_dim])

        # generate action logits considering retrieved memory and working memory
        memory_logits = self.selector(encoded_WM, retrieved_action) # (batch, LongTermMemory_size)
        assert memory_logits.size() == torch.Size([batch_size, self.hparams.LongTermMemory_size])
        
        # Brings the embedding corresponding to the action through the selected memory.
        retrieved_action = self.long_term_memory.retrieve_from_logits(memory_logits) # (batch, sentence_emb_dim)
        assert retrieved_action.size() == torch.Size([batch_size, self.sentence_emb_dim])
        
        # decode sequence considering retrieved_action and working memory
        decoder_output = self.decoder(encoded_WM, retrieved_action, action_label) # (batch, seq_len, sentence_emb_dim)
        assert decoder_output['last_hidden_state'].size() == torch.Size([batch_size, self.hparams.max_seq_length, self.sentence_emb_dim])

        return decoder_output
    
    def feature_preprocess(self,x):
        # x : (batch, max_dialogue_length, sentence_emb_dim)
        # WM_size + 1 + 1 = max_dialogue_length
        # WM, state, action, next_state = dialogue[:7], dialogue[-3], dialogue[-2], dialogue[-1]
        #encoded_WM (batch, W.M.length, sentence_emb_dim)
        #encoded_state (batch, sentence_emb_dim)
        #encoded_action (batch, sentence_emb_dim)
        #encoded_next_state (batch, sentence_emb_dim)

        return x[:,:self.hparams.WM_size,:], x[:,-3,:], x[:,-2,:], x[:,-1,:]


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward

        # sentence encode
        x, action_label = self.encoder(batch) # (batch, max_dialogue_length, sentence_emb_dim)
        batch_size = len(x)
        assert x.size()==torch.Size([batch_size, self.hparams.WM_size+2, self.sentence_emb_dim])

        # readjustment encoded batch
        encoded_WM, encoded_state, encoded_action, encoded_next_state = self.feature_preprocess(x) 

        # extract reward from small pretrained sentiment analisis model ex) thank you! +1, i don't understand.. -.5
        reward = self.sentiment_model(encoded_next_state).detach() # (batch)
        assert reward.size() == torch.Size([batch_size, 1])

        # save the memory by considering the reward.
        self.long_term_memory.save_memory(encoded_state, encoded_action, reward)
        
        # retrieve memory and select similar one
        retrieved_action = self.long_term_memory(encoded_state) # (batch,hidden)
        assert retrieved_action.size() == torch.Size([batch_size, self.sentence_emb_dim])

        # generate action logits considering retrieved memory and working memory
        memory_logits = self.selector(encoded_WM, retrieved_action) # (batch, LongTermMemory_size)
        assert memory_logits.size() == torch.Size([batch_size, self.hparams.LongTermMemory_size])
        
        # Brings the embedding corresponding to the action through the selected memory.
        retrieved_action = self.long_term_memory.retrieve_from_logits(memory_logits) # (batch, sentence_emb_dim)
        assert retrieved_action.size() == torch.Size([batch_size, self.sentence_emb_dim])
        
        # decode sequence considering retrieved_action and working memory
        decoder_output = self.decoder(encoded_WM, retrieved_action, action_label) # (batch, seq_len, sentence_emb_dim)
        assert decoder_output['last_hidden_state'].size() == torch.Size([batch_size, self.hparams.max_seq_length, self.sentence_emb_dim])


        # The memory most similar to action is used as a label.
        label_index = self.long_term_memory.retrieve_index(encoded_action).detach() # (batch)
        assert label_index.size() == torch.Size([batch_size])

        # compute policy_loss
        policy_loss = self.selector.policy_loss(encoded_WM,
                                                retrieved_action,
                                                encoded_state,
                                                label_index,
                                                reward,
                                                encoded_next_state,
                                                memory_logits.gather(1,label_index.unsqueeze(-1))) # (batch)

        # extract action from batch and compute encoder-decoder loss
        # prediction : (batch*seq,vocab), label : (batch*seq)
        e_d_loss = self.encoder_decoder_loss(decoder_output['last_hidden_state'], action_label["input_ids"]) # (batch)
        
        loss =  policy_loss + e_d_loss

        self.log('train_loss', loss)
        return loss

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