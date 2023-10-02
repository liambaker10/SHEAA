import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd

def load_model(model_name):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_summaries(model, tokenizer, input_encodings, batch_size=4):
    # Generate summaries using the model (batch processing)
    generated_summaries = []
    for i in range(0, len(input_encodings["input_ids"]), batch_size):
        batch_input_encodings = {
            key: val[i : i + batch_size] for key, val in input_encodings.items()
        }
        summary_ids = model.generate(
            **batch_input_encodings,
            max_length=100,
            num_beams=4,
            early_stopping=True,
        )
        batch_summaries = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids
        ]
        generated_summaries.extend(batch_summaries)
        print(f"Sample {i+1}/{len(input_encodings['input_ids'])} completed")
    return generated_summaries

def preprocess_data(input_texts, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(
        input_texts,
        padding="longest",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    return input_encodings

def calculate_rouge_scores(generated_summaries, target_summaries):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
    rouge_scores = []
    for generated_summary, target_summary in zip(generated_summaries, target_summaries):
        processed_generated = generated_summary.lower()
        processed_target = target_summary.lower()
        scores = scorer.score(processed_generated, processed_target)
        rouge_scores.append({'Example': f'Example {len(rouge_scores)+1}', 'ROUGE-1': scores['rouge1'].fmeasure, 'ROUGE-2': scores['rouge2'].fmeasure})
    return rouge_scores

def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=['Example', 'ROUGE-1', 'ROUGE-2'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def evaluate_model(model_name, dataset_name, num_samples, output_file, batch_size=4):
    model, tokenizer = load_model(model_name)
    dataset = load_dataset("wikitext", dataset_name,split='test')

    input_texts = dataset["text"][:num_samples]
    input_encodings = preprocess_data(input_texts, tokenizer)
    
    generated_summaries = generate_summaries(model, tokenizer, input_encodings, batch_size)
    target_summaries = input_texts  # Using input texts as target summaries for WikiText dataset

    rouge_scores = calculate_rouge_scores(generated_summaries, target_summaries)
    save_results_to_csv(rouge_scores, output_file)

# Example usage:
model_name = "facebook/bart-large-cnn"
dataset_name = "wikitext-2-raw-v1"  # Pass the appropriate dataset name as a parameter
num_samples = 10
output_file = "evaluation_results.csv"
evaluate_model(model_name, dataset_name, num_samples, output_file)