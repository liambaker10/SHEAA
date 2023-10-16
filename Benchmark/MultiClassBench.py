""" MultiClassBench because it makes use of a separate class for the dataset and a
    separate class for the actual Benchmark """
import datetime
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from datasets import load_dataset


class Processed_Dataset:

    def __init__(self, dataset_name, num_samples=None):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.input_texts, self.target_texts = self.preprocess_data()

    def preprocess_data(self):
        if self.dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            if self.num_samples is None:
                self.num_samples = len(dataset["test"]["article"])
                
            input_texts = dataset["test"]["article"][:self.num_samples]
            target_texts = dataset["test"]["highlights"][:self.num_samples]
            
        elif self.dataset_name == "bookcorpus":
            dataset = load_dataset("bookcorpus", split='train')
            if self.num_samples is None:
                self.num_samples = len(dataset)
            
            input_texts = dataset["text"][:self.num_samples]
            # For BookCorpus, input and target are the same
            target_texts = input_texts
        elif self.dataset_name == "xsum":
            dataset = load_dataset("xsum", split='test')
            if self.num_samples is None:
                self.num_samples = len(dataset["document"])
            input_texts = dataset["document"][:self.num_samples]
            target_texts = dataset["summary"][:self.num_samples]
        else:
            raise ValueError("Unsupported dataset type.")
        print("Data Preprocessing Complete")
        return input_texts, target_texts

class Benchmark:
    def __init__(self, model_name, dataset, batch_size=4):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        print(f"Tokenizer loaded: {type(self.tokenizer)}")

        if "gpt2" in model_name.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        elif "bart" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        else:
            raise ValueError("Unsupported model type.")
            
        print(f"Model loaded: {type(self.model)}")
        
        # Generate the output file name based on model_name, dataset_name, and current date and time
        current_time = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M")
        self.output_file = f"rouge_{dataset.dataset_name}_{current_time}.csv"


    def generate_summaries(self):
        input_encodings = self.tokenizer.batch_encode_plus(
            self.dataset.input_texts,
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        input_encodings = {key: val.to(self.device) for key, val in input_encodings.items()}

        generated_summaries = np.array([], dtype=str)
        for i in range(0, len(input_encodings["input_ids"]), self.batch_size):
            batch_input_encodings = {
                key: val[i: i + self.batch_size].to(self.device)
                for key, val in input_encodings.items()
            }
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **batch_input_encodings,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                )
            batch_summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            generated_summaries = np.concatenate((generated_summaries, batch_summaries), axis=None)
            
            print(f"Sample {i+1}/{len(input_encodings['input_ids'])} completed")
            
        return generated_summaries

    def calculate_rouge_scores(self):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
        rouge_scores = np.empty((len(self.generated_summaries), 2), dtype=object)  # 2 columns for 'ROUGE-1', 'ROUGE-2'

        for idx, (generated_summary, target_summary) in enumerate(zip(self.generated_summaries, self.dataset.target_texts)):
            scores = scorer.score(generated_summary.lower(), target_summary.lower())
            rouge_scores[idx] = [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure]

        return rouge_scores

    def create_directory(self):
        # Create the Results directory if it doesn't exist
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Create a directory inside Results named after model_name
        model_dir = os.path.join(results_dir, self.model_name.replace('/', '_'))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir
        
    def save_results_to_csv(self, results):

        model_directory = self.create_directory()
        # Save the results CSV file inside the model_name directory
        output_path = os.path.join(model_directory, self.output_file)
        df = pd.DataFrame(results, columns=['ROUGE-1', 'ROUGE-2'])
        df.to_csv(output_path, index=True, index_label='Example')
        print(f"Results saved to {output_path}")

    def run_evaluation(self):
        self.generated_summaries = self.generate_summaries()
        rouge_scores = self.calculate_rouge_scores()
        self.save_results_to_csv(rouge_scores)
        
        
        
class DebugBenchmark(Benchmark):
    def save_results_to_csv(self, results):
        
        model_directory = super().create_directory()

        self.output_file = self.output_file[:len(self.output_file)-4] + "_debug.csv"
        output_path = os.path.join(model_directory, self.output_file)
        
        # Prepare additional data for extra columns
        input_texts = self.dataset.input_texts
        target_texts = self.dataset.target_texts
        generated_summaries = self.generated_summaries
        
        # Create a DataFrame with additional columns
        df = pd.DataFrame(results, columns=['ROUGE-1', 'ROUGE-2'])
        df['Input Text'] = input_texts
        df['Target Text'] = target_texts
        df['Generated Summary'] = generated_summaries
        
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=True, index_label='Example')
        print(f"Results saved to {output_path}")

    def run_evaluation(self):
        self.generated_summaries = super().generate_summaries()
        rouge_scores = super().calculate_rouge_scores()
        self.save_results_to_csv(rouge_scores)

    

# Example usage:
# Currently supported models: facebook/bart-large-cnn and gpt2
MODEL_NAME = "facebook/bart-large-cnn"
# Supported Datasets: cnn_dailymail, bookcorpus, and xsum
DATASET_NAME = "xsum"
NUM_SAMPLES = 10

# Create dataset instance
DATASET = Processed_Dataset(DATASET_NAME)

# Create benchmark instance and run evaluation
benchmark = DebugBenchmark(MODEL_NAME, DATASET, batch_size=5)
benchmark.run_evaluation()