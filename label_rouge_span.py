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
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize



# os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="cnn_dailymail")
parser.add_argument("--split", type=str, default='train', help='which split of dataset')
parser.add_argument("--shard", type=int, default=128, help="divide the dataset into")
parser.add_argument("--index", type=int, default=0, help="which partition")
parser.add_argument("--dataPath", type=str, default='cnn_dailymail', help='path of files to process')
parser.add_argument("--save", type=str, default='cnn_dailymail_rouge_span', help='path to save processed data')
# parser.add_argument('--shard', action="store_true", help='use DnCNN as reference?')
# parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

opt = parser.parse_args()

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def tokenize(e):
	article = tokenizer(e['article'], truncation=True)
	return article

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

table_ = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def obj_func_single(inputs, variables):
	prediction = tokenizer.decode(inputs['input_ids'][variables[0]:variables[1]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	# prediction = ' '.join(prediction.translate(table_).split())
	ROUGE = scorer.score(inputs['highlights'], prediction)
	return ROUGE['rouge1'].fmeasure * ROUGE['rouge2'].fmeasure * ROUGE['rougeL'].fmeasure

def obj_func(inputs, x):
	return - np.array(list(map(lambda b: obj_func_single(inputs, b), x)))

class MyProblem(Problem):

	def __init__(self, inputs):
		self.inputs = inputs
		super().__init__(n_var=2, n_obj=1, n_constr=1, xl=1, xu=len(self.inputs['input_ids']), type_var=int)

	def _evaluate(self, x, out, *args, **kwargs):
		out["F"] = obj_func(self.inputs, x)
		out["G"] = x[:, 0] - x[:, 1]
		
method = get_algorithm("ga",
                       pop_size=20,
                       sampling=get_sampling("int_random"),
                       crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                       mutation=get_mutation("int_pm", eta=3.0),
                       eliminate_duplicates=True,
                       )

def span(e):
	res = minimize(MyProblem(e),
		       method,
		       termination=('n_gen', 40))
	e['labels'] =  res.X   
	e['rouge'] = - res.F 
	return e

def main():
# 	cnn_dailymail = load_from_disk(opt.dataPath)
	
# 	for k in cnn_dailymail:
# 		if k == 'test':

# 			cnn_dailymail[k] = cnn_dailymail[k].map(tokenize, batched=True)

# 			cnn_dailymail[k].set_format(type = 'numpy', columns=['input_ids', 'highlights'])

# 			cnn_dailymail[k] = cnn_dailymail[k].map(span, batched=False)
		
# # 	cnn_dailymail.remove_columns_(['attention_mask'])

# 	cnn_dailymail.save_to_disk(opt.save)
	
	raw_data = load_from_disk(opt.dataPath)[opt.split].shard(opt.shard, opt.index)	
	tokenized_data = raw_data.map(tokenize, batched=True)
	tokenized_data.set_format(type = 'numpy', columns=['input_ids', 'highlights'])
	labelled_data = tokenized_data.map(span, batched=False)	
	labelled_data.remove_columns_(['article', 'attention_mask'])
	labelled_data.save_to_disk(f"{opt.save}/{opt.split}_{opt.shard}_{opt.index}")


if __name__ == "__main__":
	main()
