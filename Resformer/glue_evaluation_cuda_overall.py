# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount("/content/gdrive/")
# root_path = "/content/gdrive/MyDrive/"
# %cd "/content/gdrive/MyDrive/resformer"
import copy
import argparse
import torch
import torch.nn as nn
import random
import requests
import datasets
import transformers
import pandas as pd
import terrorch
from terrorch import Injector
import time
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
print(transformers.__version__)

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.Tensor(predictions)
    if task != "stsb":
        predictions = torch.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def evaluation(input_model):
    evaluator = Trainer(input_model, eval_dataset=encoded_dataset[validation_key], tokenizer=tokenizer, compute_metrics=compute_metrics)
    return evaluator.evaluate()
    


GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

parser = argparse.ArgumentParser()

parser.add_argument('--task', metavar='N', type=str)

parser.add_argument('--model', metavar='N', type=str)

parser.add_argument('--ber', metavar='N', type=str)

parser.add_argument('--seed', metavar='N', type=int)

parser.add_argument('--component', metavar='N', type=str)

args = parser.parse_args()

task, model_and_task, bit_error_rate, seed = args.task, args.model, args.ber, args.seed

actual_task = "mnli" if task == "mnli-mm" else task

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

url_test = 'https://huggingface.co/' + model_and_task + '/resolve/main/config.json'

if requests.head(url_test).status_code != 404:

    model_inj = AutoModelForSequenceClassification.from_pretrained(model_and_task, num_labels=num_labels)

else:
    
    print("ERROR: selected model don't exist")
    
    exit()

dataset = load_dataset("glue", actual_task)

metric = load_metric('glue', actual_task)

tokenizer = AutoTokenizer.from_pretrained(model_and_task)

sentence1_key, sentence2_key = task_to_keys[task]

encoded_dataset = dataset.map(preprocess_function, batched=True)

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

# model_inj.to(device)

with torch.no_grad():
    # eval_result = evaluation(model_test0)
    # print('................................................')
    # print(bit_error_rate, eval_result)
    # print('................................................')

    eval_result = evaluation(model_test1)
    print('................................................')
    print(bit_error_rate, eval_result)
    print('................................................')

    eval_result = evaluation(model_test2)
    print('................................................')
    print(bit_error_rate, eval_result)
    print('................................................')