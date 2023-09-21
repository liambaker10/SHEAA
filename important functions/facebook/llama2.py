# Use a pipeline as a high-level helper
from transformers import pipeline


# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
def generate_responses(mask):
    unmasker = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    outputs = unmasker("Hello, how are you?")

    responses = [output["sequence"] for output in outputs]
    formatted_response = " ".join(responses)
    return formatted_response


output = generate_responses("Hi I'm Liam")
print(output)
