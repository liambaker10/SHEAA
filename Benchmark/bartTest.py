from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Prompt text
prompt = "Once upon a time, in a land far, far away,"

# Encode the input text to numerical tokens
input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

# Generate text based on the input
output = model.generate(input_ids, max_length=100, num_beams=4, length_penalty=2.0, early_stopping=True, min_length=20)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
