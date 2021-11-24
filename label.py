import os
import argparse

import time
import torch
import string
import numpy as np

from transformers import RobertaTokenizerFast
from datasets import load_dataset, ReadInstruction, load_from_disk
from rouge_score import rouge_scorer

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize


# os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="cnn_dailymail")
parser.add_argument("--split", type=str, default='train', help='which split of dataset')
parser.add_argument("--shard", type=int, default=128, help="divide the dataset into")
parser.add_argument("--index", type=int, default=0, help="which partition")
parser.add_argument("--dataPath", type=str, default='/content/drive/My Drive/Colab Notebooks/cnn_dailymail', help='path of files to process')
parser.add_argument("--save", type=str, default='/content/drive/My Drive/Colab Notebooks/cnn_dailymail_subseq', help='path to save processed data')
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

def enlarge(ids):
	A = set(ids)
	B = set().union(*[interchangeable[a] for a in A])
	return B, B - A

def punkts():
	punkt = set()
	G = tokenizer.convert_ids_to_tokens(1437)

	for k,v in tokenizer.vocab.items():
		if not any(letter.isalnum() for letter in k.replace(G, '')):
			punkt.add(v)

	return punkt

punctuation = punkts()

def confine(input_ids, highlights_ids):
	W = enlarge(highlights_ids)[0] - {1}

	variables = []
	labels = []
	for i, x in enumerate(input_ids):
		if x in punctuation:
			labels.append(1)
		else:
			labels.append(0)
			if x in W:
				variables.append(i)

	return labels, variables

def tokenize(e):
	article = tokenizer(e['article'], truncation=True)
	highlights = tokenizer(e['highlights'], truncation=True)
	article['highlight_ids'] = highlights['input_ids']
	labels, confined = zip(*list(map(confine, article['input_ids'], article['highlight_ids'])))
	article['confined'] = list(confined)
	article['labels'] = list(labels)
	return article


def obj_func_single(inputs, variables):
	mask = inputs['labels'].astype(bool)
	mask[inputs['confined']] = variables
	prediction = tokenizer.decode(inputs['input_ids'][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	# prediction = ' '.join(prediction.translate(table_).split())
	ROUGE = scorer.score(inputs['highlights'], prediction)
	return ROUGE['rouge1'].fmeasure * ROUGE['rouge2'].fmeasure * ROUGE['rougeL'].fmeasure

def obj_func(inputs, x):
	return - np.array(list(map(lambda b: obj_func_single(inputs, b), x)))

class MyProblem(Problem):

	def __init__(self, inputs):
		self.inputs = inputs
		super().__init__(n_var=len(self.inputs['confined']), n_obj=1, n_constr=0, xl=0, xu=1, type_var=int)

	def _evaluate(self, x, out, *args, **kwargs):
		out["F"] = obj_func(self.inputs, x)

def highlight(e):
	e['labels'][e['confined']] = minimize(MyProblem(e),
										 algorithm,
										 ('n_gen', 90),
										 verbose=False).X.astype(int)
	return e

def main():
	useful = lambda x: len(x['article']) > len(x['highlights']) + 500

	raw_data = load_from_disk(opt.dataPath)[opt.split].filter(useful).shard(opt.shard, opt.index)

	tokenized_data = raw_data.map(tokenize, batched=True)

	print(f"Number of training examples: {len(tokenized_data)}")

	tokenized_data.set_format(type = 'numpy', columns=['input_ids', 'labels', 'confined', 'highlights'])

	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

	table_ = str.maketrans(string.punctuation, ' '*len(string.punctuation))

	algorithm = GA(
		pop_size=28,
		sampling=get_sampling("bin_random"),
		crossover=get_crossover("bin_hux"),
		mutation=get_mutation("bin_bitflip"),
		eliminate_duplicates=True)

	labelled_data = tokenized_data.map(highlight, batched=False)

	labelled_data.save_to_disk(f"{opt.save}/{opt.split}_{opt.shard}_{opt.index}")


if __name__ == "__main__":
	main()
