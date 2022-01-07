# To create tags for each token in CNN/Dailymail dataset
We choose RobertaTokenizerFast, and truncate at first 512 tokens

Install necessary packages
```
pip install datasets > /dev/null
pip install transformers > /dev/null
pip install rouge-score > /dev/null
pip install pymoo >/dev/null
```
prepare Huggingface `cnn_dailymail` dataset (load from official (Huggingface)[https://huggingface.co/datasets/cnn_dailymail] website, and use `save_to_disk` to save the dataset (dict) into a folder on your disk), and get this repository ready

```
python label_ngrams.py #--shard 10000 --index 0
```

## Align sharded dataset with original one

```python
import numpy as np
import matplotlib.pyplot as plt

from transformers import RobertaTokenizerFast, RobertaForTokenClassification

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


from datasets import load_from_disk, concatenate_datasets

cnn_dailymail = load_from_disk('cnn_dailymail')
cnn_dailymail_rouge_span_test = concatenate_datasets([load_from_disk(f'cnn_dailymail_rouge_span_{i}')['test'] for i in range(3)])
```

For spans
```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def rouge(e):
    ROUGE = scorer.score(e['highlights'], tokenizer.decode(e['input_ids'][e['labels'][0]:e['labels'][1]], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    e['ROUGE'] = [[ROUGE['rouge1'].precision, ROUGE['rouge2'].precision, ROUGE['rougeL'].precision],
            [ROUGE['rouge1'].recall, ROUGE['rouge2'].recall, ROUGE['rougeL'].recall],
            [ROUGE['rouge1'].fmeasure, ROUGE['rouge2'].fmeasure, ROUGE['rougeL'].fmeasure]]
    return e
```

For subsequences
```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def rouge(e):
    ROUGE = scorer.score(e['highlights'], tokenizer.decode(e['input_ids'][e['labels']%2 == 1], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    e['ROUGE'] = [[ROUGE['rouge1'].precision, ROUGE['rouge2'].precision, ROUGE['rougeL'].precision],
            [ROUGE['rouge1'].recall, ROUGE['rouge2'].recall, ROUGE['rougeL'].recall],
            [ROUGE['rouge1'].fmeasure, ROUGE['rouge2'].fmeasure, ROUGE['rougeL'].fmeasure]]
    return e
```

```python
def table(raw_dataset, *columns):
    return {x[0]:dict(zip(columns, x[1:])) for x in zip(raw_dataset['id'], *[raw_dataset[column] for column in columns])}
```
```python
tab = table(cnn_dailymail_rouge_span_test, 'labels', 'input_ids')

def align(e, tab = tab):
    e.update(tab[e['id']])
    return e
```
```python
newtest = cnn_dailymail['test'].map(align, batched = False).map(rouge, batched=False)
```
```python
from scipy.ndimage import gaussian_filter1d as gf

np.median(np.array(newtest['ROUGE']), axis =0)

plt.plot(gf([x[0][2] for x in newtest['ROUGE']], 10))
plt.plot(gf([x[1][2] for x in newtest['ROUGE']], 10))
plt.plot(gf([x[2][2] for x in newtest['ROUGE']], 10))
```

