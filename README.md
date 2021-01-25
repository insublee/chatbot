# chatbot

Memory augmented reinforce learning chatbot

## Installation

1. Prerequisites

```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```


2. Clone repository

```bash
git clone https://github.com/2B3E/plato.git
```



3. Change directory

```bash
cd chatbot
```


4. Make a virtual environment (if needed)

```bash
python3 -m venv ./venv
```

5. Activate the virtual environment (if needed)

```bash
source ./venv/bin/activate
```

6. Dependencies

```bash
pip install -r requirements.txt
```

7. "Editable" install


```bash
pip install -e .
```

## Usage

```python
import chatbot

chatbot = chatbot.~~~()

utterance = "hi, my name is insub. who are you?"

response = chatbot(utterance=utterance)

print(response)
```
