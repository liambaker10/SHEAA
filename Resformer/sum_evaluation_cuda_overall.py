import copy
import nltk
import torch
import torch.nn as nn
import random
import argparse
import datasets
import transformers
import numpy as np
import pandas as pd
import terrorch
from terrorch import Injector
import time
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
print(transformers.__version__)

nltk.download('punkt')

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

parser = argparse.ArgumentParser()

parser.add_argument('--task', metavar='N', type=str)

parser.add_argument('--model', metavar='N', type=str)

parser.add_argument('--ber', metavar='N', type=float)

parser.add_argument('--seed', metavar='N', type=str)

args = parser.parse_args()

task, model_and_task, bit_error_rate, seed = args.task, args.model, args.ber, args.seed

task, model_and_task, bit_error_rate, seed = 'xsum', 'Katsie011/t5-small-finetuned-xsum', 1e-5, 0

raw_datasets = load_dataset(task)

metric = load_metric("rouge")

prefix = "summarize: "

max_input_length = 1024

max_target_length = 128
    
tokenizer = AutoTokenizer.from_pretrained(model_and_task)

"""If you are using one of the five T5 checkpoints we have to prefix the inputs with "summarize:" (the model can also translate and it needs the prefix to know which task it has to perform)."""

tokenized_datasets = raw_datasets.map(preprocess_function, batched = True)

model_inj = AutoModelForSeq2SeqLM.from_pretrained(model_and_task)

device = torch.device('cpu')

with torch.no_grad():

    injector = Injector(param_names = ['weight', 'bias'], p = float(bit_error_rate), device = device, seed = seed, verbose = True)

    injector.profiling(model_inj)

    model_test0 = copy.deepcopy(model_inj)
    
    injector.inject(model_inj)

    model_test1 = copy.deepcopy(model_inj)

    injector.correct(model_inj)

    model_test2 = copy.deepcopy(model_inj)

device = torch.device('cuda')

model_test0.to(device)

model_test1.to(device)

model_test2.to(device)

batch_size = 32

def eval(input_model):
    args = Seq2SeqTrainingArguments(f"output", predict_with_generate = True,)
    evaluator = Seq2SeqTrainer(input_model, args, eval_dataset=tokenized_datasets["validation"], data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_metrics)
    return evaluator.evaluate()

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_test0)

with torch.no_grad():
    eval_result = eval(model_test0)
    print('................................................')
    print('Error free model', bit_error_rate, eval_result)
    print('................................................')

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_test1)

with torch.no_grad():
    eval_result = eval(model_test1)
    print('................................................')
    print('Error injected model', bit_error_rate, eval_result)
    print('................................................')

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_test2)

with torch.no_grad():
    eval_result = eval(model_test2)
    print('................................................')
    print('Error corrected model', bit_error_rate, eval_result)
    print('................................................')