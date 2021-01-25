# plato

Dialogue engine for language education

## Installation

1. Prerequisites

```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```


2. Clone repository

for SSH:

```bash
git clone git@github.com:2B3E/plato.git
```

otherwise:


```bash
git clone https://github.com/2B3E/plato.git
```



3. Change directory

```bash
cd plato
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
import plato


context = r"""
I love food.
I see a cake.
I taste the cake.
It tastes sweet.
A cherry is on the cake.
It looks yummy.
I love food.
I see cookies.
Chocolate is on the cookies.
It looks sweet.
I taste the cookies.
They taste great.
"""

question = "What does the boy love?"

qa = plato.eng.QuestionAnswering()

answer = qa(question=question, context=context)

print(answer)
```
