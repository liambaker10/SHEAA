import transformers
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForMaskedLM


def robertaResponse(prompt, attack, sf, p):

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



    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    model = model.to(device)

    error_inject(model, attack, sf, p)

    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Find the position of the masked token
    masked_index = token_ids.index(tokenizer.mask_token_id)

    # Convert token IDs to tensors
    input_tensor = torch.tensor([token_ids]).to(device)

    # Pass the input tensor through the RoBERTa model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the predictions for the masked token
    predictions = outputs.logits[0, masked_index]
    predicted_token_ids = torch.argmax(predictions, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_ids.tolist())

    # Replace the masked token in the input text with the predicted token
    output_text = prompt.replace("<mask>", predicted_token)

    # Print the output text
    return output_text
