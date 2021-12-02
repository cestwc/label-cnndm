import os
import argparse

import time
import torch
import numpy as np
from collections import Counter

from transformers import RobertaTokenizerFast
from datasets import load_dataset, load_from_disk


# os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="cnn_dailymail")
# parser.add_argument("--split", type=str, default='train', help='which split of dataset')
# parser.add_argument("--shard", type=int, default=128, help="divide the dataset into")
# parser.add_argument("--index", type=int, default=0, help="which partition")
parser.add_argument("--dataPath", type=str, default='cnn_dailymail', help='path of files to process')
parser.add_argument("--save", type=str, default='/content/drive/My Drive/Colab Notebooks/cnn_dailymail_ngrams', help='path to save processed data')
# parser.add_argument('--shard', action="store_true", help='use DnCNN as reference?')
# parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

opt = parser.parse_args()

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


def alike():
	vocab = tokenizer.get_vocab()
	tokens = {k:{v} for k, v in vocab.items()}
	s = tokenizer.convert_ids_to_tokens(1437)
	for k in tokens:
		if not k.islower():
			if s in k:
				naked = k.replace(s, '')
				if naked in tokens:
					tokens[k].add(vocab[naked])
					tokens[naked] = tokens[k]
			else:
				naked = k
			lowercase = naked.lower()
			if lowercase in tokens:
				tokens[k].add(vocab[lowercase])
				tokens[lowercase] = tokens[k]

	return {vocab[k]:v for k, v in tokens.items()}

interchangeable = alike()

def punkts():
	punkt = set()
	G = tokenizer.convert_ids_to_tokens(1437)

	for k,v in tokenizer.vocab.items():
		if not any(letter.isalnum() for letter in k.replace(G, '')):
			punkt.add(v)

	return punkt

punctuation = punkts()

def unigrams(e):
	input_ids = e['input_ids']
	highlights_ids = e['highlights_ids']
	
	bag = dict(Counter(highlights_ids))
	vocab = set(input_ids)
	input_ids = np.array(input_ids)
	
	e['unigrams'] = np.zeros_like(input_ids)

	for k, v in bag.items():
		if k in vocab:
			e['unigrams'][input_ids == k] += v
		else:
			for a in interchangeable[k]:
				if a in vocab:
					e['unigrams'][input_ids == a] += v
					break
	return e

def bigrams(e):
	input_ids = e['input_ids']
	highlights_ids = e['highlights_ids']

	bag = dict(Counter(list(zip(highlights_ids[:-1], highlights_ids[1:]))))
	scope = set(list(zip(input_ids[:-1], input_ids[1:])))
	input_ids = np.array(input_ids)

	e['bigrams'] = np.zeros_like(input_ids)
	candidates = np.stack((input_ids, np.roll(input_ids, -1)))

	for k, v in bag.items():
		if k in scope:
			e['bigrams'][np.all(candidates == np.array([list(k)]).T, axis=0)] += v
		else:
			for a in interchangeable[k[0]]:
				for b in interchangeable[k[1]]:
					if a in scope:
						e['bigrams'][np.all(candidates == np.array([[a, b]]).T, axis=0)] += v
						break
	return e

def tokenize(e):
	article = tokenizer(e['article'], truncation=True)
	highlights = tokenizer(e['highlights'], truncation=True)
	article['highlight_ids'] = highlights['input_ids']
	return article

def main():
	cnn_dailymail = load_from_disk(opt.dataPath)
	
	for k in cnn_dailymail:

		cnn_dailymail[k] = cnn_dailymail[k].map(tokenize, batched=True)

		cnn_dailymail[k] = cnn_dailymail[k].map(unigrams, batched=False)

		cnn_dailymail[k] = cnn_dailymail[k].map(bigrams, batched=False)

	cnn_dailymail.save_to_disk(opt.save)


if __name__ == "__main__":
	main()
