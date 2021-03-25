import pytorch_lightning as pl
from args import args
from chatbot.data import DataModule
from chatbot import Chatbot

print('#'*10, 'model download & init','#'*10)
hparams = args()
chatbot = Chatbot(hparams)

print('#'*10, 'data download & preparing','#'*10)
dm = DataModule(hparams)
dm.prepare_data()
dm.setup('fit')

print('#'*10, 'data check','#'*10)
batch = next(iter(dm.train_dataloader()))
for i in batch.keys():
    print(i, batch[i].size())

print('#'*10, 'forward test','#'*10)
forward_out = chatbot(**batch)
print(forward_out.keys()) # odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])

print('#'*10, 'training step test','#'*10)
loss = chatbot.training_step(batch,0)
print(loss) # tensor(14.2616, grad_fn=<AddBackward0>)

print('#'*10, 'validation step test','#'*10)
loss = chatbot.validation_step(batch,0)
print(loss) # tensor(14.2616, grad_fn=<AddBackward0>)

print('#'*10, 'generation test1. no input','#'*10)
outputs = chatbot.model.generate(max_length=40)
print("Generated:", dm.tokenizer.decode(outputs[0], skip_special_tokens=True))

print('#'*10, "generation test2. input : 'hello my name is insub, and'",'#'*10)
input_text = "hello my name is insub, and"
tokenized_text = dm.tokenizer([input_text], return_tensors="pt").input_ids
outputs = chatbot.model.generate(input_ids=tokenized_text, max_length=40)
print("Generated:", dm.tokenizer.decode(outputs[0], skip_special_tokens=True))

print('#'*10, 'trainer loading and fit','#'*10)
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(chatbot, dm)

print('#'*10, 'generation test3. after training no input','#'*10)
outputs = chatbot.model.generate(max_length=40)
print("Generated:", dm.tokenizer.decode(outputs[0], skip_special_tokens=True))

print('#'*10, "generation test4. after training input : 'hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .'",'#'*10)
input_text = "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape ."
tokenized_text = dm.tokenizer([input_text], return_tensors="pt").input_ids
outputs = chatbot.model.generate(input_ids=tokenized_text, max_length=40)
print("Generated:", dm.tokenizer.decode(outputs[0], skip_special_tokens=True))