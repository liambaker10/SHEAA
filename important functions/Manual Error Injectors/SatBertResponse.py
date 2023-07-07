import transformers
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer


def bertResponse(prompt, attack, sf, p):

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




    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = model.to(device)

    error_inject(model, attack, sf, p)

    # Input text
    input_text = prompt


    input_tokens = tokenizer.tokenize(input_text)
    masked_index = input_tokens.index("[MASK]")
    # Tokenize the input text


    # Add special tokens to the input tokens
    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]

    # Encode the input text
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    # Get the predictions for the masked token
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

        # Ensure the predictions tensor has the expected shape
        assert (
            predictions.shape[1] >= masked_index + 1
        ), "Masked index is out of bounds."

        # Get the predicted token for the masked position
        predicted_token_index = torch.argmax(
            predictions[0, masked_index + 1]
        ).item()  # Add 1 to masked_index due to [CLS] token
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_index)


    # Decode the generated output
    return predicted_token

