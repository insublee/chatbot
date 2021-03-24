# chatbot

Memory augmented reinforce learning chatbot

![image](https://www.notion.so/insub/Presentation-229f5d9967d54b758fcfe42c9dfe7c80#c028ef50e5ec440b8a0ccf2b9ffec9cc)


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
from args import args

hparams = args()

dm = DataModule(hparams)
dm.prepare_data()
dm.setup('fit')

chatbot = Chatbot(hparams)
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(chatbot, dm)

```
