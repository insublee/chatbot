# chatbot

Memory augmented reinforce learning chatbot

## Installation

1. Clone repository

```bash
git clone https://github.com/insublee/chatbot.git
```

2. Change directory

```bash
cd chatbot
```

3. install Dependencies

```bash
pip install -r requirements.txt
```


## Usage

```python
from chatbot import chatbot

chatbot = Chatbot()

utterance = "hi, my name is insub. who are you?"

response = chatbot(utterance)

```

## fine-tunning

```python
import pytorch_lightning as pl
from chatbot.data import DataModule
from chatbot import Chatbot

dm = DataModule("facebook/bart-base",use=0.004)
dm.prepare_data()
dm.setup('fit')

chatbot  = Chatbot()

trainer = pl.Trainer(max_epochs=2)
trainer.fit(chatbot, dm.train_dataloader()) # 1epoch 8시간정도

```
