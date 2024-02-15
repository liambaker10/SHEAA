# My commentsa have -SB on them the rest are Ruixuans or GPTS from the copied llama2chat file
from transformers import LlamaTokenizerFast, LlamaForCausalLM
import torch
import torch.cuda
from huggingface_hub import login
from pytei import Injector, Defender

#login() # If yo are running on your machine uncomment this and get your hugging face user access token from https://huggingface.co/settings/tokens -SB
# Once you login once you don't need to do it again

#Downloading the model locally - SB
tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


# Create an instance of the Injector class
target_path = "C:\\Users\\satra\\OneDrive\\Desktop\\New SHEAA\\SHEAA\Benchmark\\targetsllama.txt"
injector = Injector(
    target_path=target_path,
    p=1e-10,  # Probability of error injection
    dtype=torch.float,
    device=torch.device('cpu' if torch.cuda.is_available() else 'cpu'),  # I forced to run on the CPU since CUDA out of memory hapens if not - SB
    verbose=True,  # Set to True to print information about error injection
    error_model='bit',  # Choose 'bit' or 'value' for error model
    mitigation='None'  # Choose 'None', 'SBP', or 'clip' for mitigation
)

injector.inject(model, use_mitigation=True)  # Use `use_mitigation=True` to apply mitigation if specified

print(f"Model loaded: {type(model)}") 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to("cpu")  # I forced to run on the CPU since CUDA out of memory hapens if not 

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
    #batch_size=1
)

# Decode and print the generated text
for seq in sequences:
    generated_text = tokenizer.decode(seq, skip_special_tokens=True)
    print(f"Result: {generated_text}")

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
