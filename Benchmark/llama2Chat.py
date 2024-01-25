from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

<<<<<<< HEAD
# Specify the cache directory
# Choose where huggingface model will be installed
cache_dir = "C:\Users\liamb\OneDrive\Research\cache"
# Set the TRANSFORMERS_CACHE environment variable
os.environ["TRANSFORMERS_CACHE"] = cache_dir
=======
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: {type(tokenizer)}")
>>>>>>> 3a5ba1bf5d197193aab35713a8971db66b98fa54

model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Model loaded: {type(model)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # You can specify the appropriate device here, like "cuda" for GPU or "cpu" for CPU
print(f"Device: {device}")


input_prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

# Encoding the input text using the tokenizer
input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(model.device)
print("Starting generation...")
# Generate text based on the input prompt
sequences = model.generate(
    input_ids,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
    batch_size=1
)

# Decode and print the generated text
for seq in sequences:
    generated_text = tokenizer.decode(seq, skip_special_tokens=True)
    print(f"Result: {generated_text}")
