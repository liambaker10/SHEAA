# If you get cuda out of memory error, lower the batch_size parameter
import torch
from torch.cuda.amp import autocast
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd

def preprocess_data(input_texts, target_texts, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(
        input_texts,
        padding="longest",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    target_encodings = tokenizer.batch_encode_plus(
        target_texts,
        padding="longest",
        truncation=True,
        max_length=100,
        return_tensors="pt",
    )
    print("Finished preprocessing")
    return input_encodings, target_encodings


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=['Example', 'ROUGE-1', 'ROUGE-2'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def evaluate_bart_model(model_name, dataset_name, num_samples, output_file):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Load the dataset
    dataset = load_dataset(dataset_name, "3.0.0")

    # Prepare the input and target texts
    input_texts = dataset["test"]["article"][:num_samples]
    target_texts = dataset["test"]["highlights"][:num_samples]

    # Preprocess data
    input_encodings, target_encodings = preprocess_data(
        input_texts, target_texts, tokenizer
    )

    # Generate summaries using the BART model (batch processing)
    generated_summaries = []
    batch_size = 4
    for i in range(0, len(input_encodings["input_ids"]), batch_size):
        batch_input_encodings = {
            key: val[i : i + batch_size].to(device)
            for key, val in input_encodings.items()
        }
        with torch.cuda.amp.autocast():
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

    # Decode target summaries
    target_summaries = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in target_encodings["input_ids"]
    ]
    print("Target summaries decoded")

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
    rouge_scores = []
    for generated_summary, target_summary in zip(generated_summaries, target_summaries):
        processed_generated = generated_summary.lower()  # Apply necessary preprocessing
        processed_target = target_summary.lower()  # Apply necessary preprocessing
        scores = scorer.score(processed_generated, processed_target)
        rouge_scores.append({'Example': f'Example {len(rouge_scores)+1}', 'ROUGE-1': scores['rouge1'].fmeasure, 'ROUGE-2': scores['rouge2'].fmeasure})

    # Save results to CSV
    save_results_to_csv(rouge_scores, output_file)


# Example usage:
model_name = "facebook/bart-large-cnn"
dataset_name = "cnn_dailymail"
num_samples = 10  # Specify the number of samples to process
output_file = "evaluation_results.csv"  # Specify the output file path
evaluate_bart_model(model_name, dataset_name, num_samples, output_file)
