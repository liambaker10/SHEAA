# Only works with Bart and cnn_dailymail dataset
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd

class Benchmark:
    def __init__(self, model_name, dataset_name, num_samples, output_file, batch_size=4):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.output_file = output_file
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        print(f"Tokenizer: {type(self.tokenizer)}")

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name, "3.0.0")

    def preprocess_data(self):
        input_texts = self.dataset["test"]["article"][:self.num_samples]
        target_texts = self.dataset["test"]["highlights"][:self.num_samples]
        input_encodings = self.tokenizer.batch_encode_plus(
            input_texts,
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        input_encodings = {key: val.to(self.device) for key, val in input_encodings.items()}
        return input_encodings, target_texts

    def generate_summaries(self, input_encodings):
        generated_summaries = []
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
            batch_summaries = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids
            ]
            generated_summaries.extend(batch_summaries)
        return generated_summaries

    def calculate_rouge_scores(self, generated_summaries, target_summaries):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
        rouge_scores = []

        for idx, (generated_summary, target_summary) in enumerate(zip(generated_summaries, target_summaries), 1):
            scores = scorer.score(generated_summary.lower(), target_summary.lower())
            rouge_scores.append({
                'Example': idx,  # Include the example number in the results
                'ROUGE-1': scores['rouge1'].fmeasure,
                'ROUGE-2': scores['rouge2'].fmeasure
            })

        return rouge_scores

    def save_results_to_csv(self, results):
        df = pd.DataFrame(results, columns=['Example', 'ROUGE-1', 'ROUGE-2'])
        df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")

    def run_evaluation(self):
        self.load_model()
        self.load_dataset()
        input_encodings, target_texts = self.preprocess_data()
        generated_summaries = self.generate_summaries(input_encodings)
        target_summaries = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_encodings["input_ids"]]
        rouge_scores = self.calculate_rouge_scores(generated_summaries, target_summaries)
        self.save_results_to_csv(rouge_scores)

# Example usage:
model_name = "facebook/bart-large-cnn"
dataset_name = "cnn_dailymail"
num_samples = 10
output_file = "evaluation_results.csv"

benchmark = Benchmark(model_name, dataset_name, num_samples, output_file)
benchmark.run_evaluation()
