from transformers import pipeline

def generate_responses(mask):
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    outputs = unmasker("Hello I'm a [MASK] model.")

    responses = [output["sequence"] for output in outputs]
    formatted_response = " ".join(responses)
    return formatted_response

output = generate_responses("professional")
print(output)






