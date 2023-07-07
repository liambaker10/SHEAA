import transformers
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration


def bartResponse(prompt, attack, sf, p):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Specify the data type you want to retrieve the number of bits for
    bitwidth = torch.finfo(dtype).bits

    def error_map(injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, scale_factor = .1, p = 1e-10) -> torch.Tensor:
        with torch.no_grad():
            error_map = (2 * torch.ones((*injectee_shape, dtype_bitwidth), dtype=torch.int, device=device)) ** torch.arange(0, dtype_bitwidth, dtype=torch.int, device=device).flip(dims=(-1, )).expand((*injectee_shape, dtype_bitwidth))

            filter = (p * nn.functional.dropout(torch.ones_like(error_map, dtype=torch.float, device=device), 1 - p)).int()

            error_map = (filter * error_map * scale_factor).sum(dim=-1).int()

        return error_map

    def error_inject(model, attack, sf, p):
        error_maps = {}

        for param_name, param in model.named_parameters():
            # Options for attacks are here, you can do weights in general bias in general
            # Then you can do specific kinds of weights/biases attn.weights, proj.weights
            if attack in param_name:# or "bias" in param_name:

                injectee_shape = param.shape

                error_maps[param_name] = error_map(injectee_shape, bitwidth, device, sf, p)

                error_fin = error_maps[param_name]

                param.data = (param.data.to(torch.int) ^ error_fin).to(torch.float)




    model_name = 'facebook/bart-large-cnn'
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Input text
    input_text = prompt

    # Tokenize the input text
    input_tokens = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt'
    )
    input_tokens = input_tokens.to(device)
    #Error injection needs a model which is gpt2 an attack name as a string sf as a scale factor to reduce errors and p to introduce randomness
    error_inject(model, attack, sf, p)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            max_length=100,  # Specify the desired maximum length of the generated text
            num_beams=5,   # Specify the number of beams for beam search
            early_stopping=True,
            no_repeat_ngram_size=2,
            num_return_sequences=1
        )



    # Decode the generated output
    generated_text = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
    return generated_text

