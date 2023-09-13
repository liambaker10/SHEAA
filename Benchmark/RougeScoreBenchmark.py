# Second Iteration of Benchmark using rouge-score to evaluate base Bart model
# Outputs csv with rouge score data

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd

# Global variables
batch_size = 4  # Adjust the batch size as needed
num_examples_to_process = 100  # Change this number to the desired limit
max_length = 100  # Adjust the max_length as needed
gradient_accumulation_steps = 4  # Adjust as needed

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BART model and tokenizer and move them to the GPU
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load the WikiText dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Create a function to generate text using the model
def generate_texts(prompts):
    generated_texts = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        input_ids = tokenizer.batch_encode_plus(batch_prompts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        
        # Move input_ids to the GPU
        input_ids = input_ids.to(device)
        
        output_ids = model.generate(input_ids.input_ids, max_length=max_length, num_return_sequences=1)
        batch_generated_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
        generated_texts.extend(batch_generated_texts)
        print(f"Processed {len(generated_texts)} of {len(prompts)} examples...")
    return generated_texts

# Initialize the Rouge scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Retrieve input prompts (text) using integer indexing
input_prompts = dataset['test']['text'][:num_examples_to_process]

# Generate text for the selected examples
generated_texts = generate_texts(input_prompts)

# Initialize Rouge score accumulator
total_rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

# Create a list to store data for each example
data = []

for input_text, generated_text in zip(input_prompts, generated_texts):
    # Calculate Rouge scores
    scores = scorer.score(input_text, generated_text)

    # Accumulate the scores
    for metric, score in scores.items():
        total_rouge_scores[metric] += score.fmeasure

    # Append data for each example
    data.append({
        'prompt': input_text,
        'answer': generated_text,
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    })

# Calculate average Rouge scores
total_examples = len(input_prompts)
average_rouge_scores = {metric: score / total_examples for metric, score in total_rouge_scores.items()}

# Print the results
print("Average Rouge Scores:")
for metric, score in average_rouge_scores.items():
    print(f"{metric}: {score:.4f}")

# Save data to a CSV file
df = pd.DataFrame(data)
df.to_csv('rouge_scores.csv', index=False)
