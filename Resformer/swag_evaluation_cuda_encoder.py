# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount("/content/gdrive/")
# root_path = "/content/gdrive/MyDrive/"
# %cd "/content/gdrive/MyDrive/resformer"
import copy
import argparse
import torch
import requests
import random
import datasets
import transformers
import numpy as np
import pandas as pd
import torch.nn as nn
import terrorch
from terrorch import Injector
import time
from datasets import load_dataset, load_metric

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
print(transformers.__version__)

batch_size = 512

parser = argparse.ArgumentParser()

parser.add_argument('--task', metavar='N', type=str)

parser.add_argument('--model', metavar='N', type=str)

parser.add_argument('--ber', metavar='N', type=float)

parser.add_argument('--seed', metavar='N', type=str)

args = parser.parse_args()

task, model_and_task, bit_error_rate, seed = args.task, args.model, args.ber, args.seed

url_test = 'https://huggingface.co/' + model_and_task + '/resolve/main/config.json'

if requests.head(url_test).status_code != 404:

    pass

else:
    
    print("ERROR: selected model don't exist")
    
    exit()

device = torch.device('cpu')

datasets = load_dataset("swag", "regular")
    
tokenizer = AutoTokenizer.from_pretrained(model_and_task,  use_fast=True)

ending_names = ["ending0", "ending1", "ending2", "ending3"]

def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

encoded_datasets = datasets.map(preprocess_function, batched=True)

model_inj = AutoModelForMultipleChoice.from_pretrained(model_and_task)

with torch.no_grad():

    injector = Injector(param_names = ['encoder'], p = float(bit_error_rate), device = device, seed = seed, verbose = True)

    injector.profiling(model_inj)

    model_test0 = copy.deepcopy(model_inj)
    
    injector.inject(model_inj)

    model_test1 = copy.deepcopy(model_inj)

    injector.correct(model_inj)

    model_test2 = copy.deepcopy(model_inj)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

device = torch.device('cuda')

model_test0.to(device)

model_test1.to(device)

model_test2.to(device)

def eval(input_model):
    evaluator = Trainer(input_model, eval_dataset=encoded_datasets["validation"], tokenizer=tokenizer, data_collator=DataCollatorForMultipleChoice(tokenizer), compute_metrics=compute_metrics)
    return evaluator.evaluate()

with torch.no_grad():
    # eval_result = eval(model_test0)
    # print('................................................')
    # print(bit_error_rate, eval_result)
    # print('................................................')

    eval_result = eval(model_test1)
    print('................................................')
    print(bit_error_rate, eval_result)
    print('................................................')

    eval_result = eval(model_test2)
    print('................................................')
    print(bit_error_rate, eval_result)
    print('................................................')