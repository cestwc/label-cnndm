# To create tags for each token in CNN/Dailymail dataset
We choose RobertaTokenizerFast, and truncate at first 512 tokens

Install necessary packages
```
pip install datasets > /dev/null
pip install transformers > /dev/null
pip install rouge-score > /dev/null
pip install pymoo >/dev/null
```
prepare Huggingface `cnn_dailymail` dataset, agd get this repository ready

```
python label_rougest_subsequence.py --shard 10000 --index 0
```
