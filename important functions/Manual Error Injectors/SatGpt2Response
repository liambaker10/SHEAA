import transformers
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


def gptResponse(prompt, attack, sf, p):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float32
    )  # Specify the data type you want to retrieve the number of bits for
    bitwidth = torch.finfo(dtype).bits

    def error_map(
        injectee_shape: tuple,
        dtype_bitwidth: int,
        device: torch.device,
        scale_factor=0.1,
        p=1e-10,
    ) -> torch.Tensor:
        with torch.no_grad():
            error_map = (
                2
                * torch.ones(
                    (*injectee_shape, dtype_bitwidth), dtype=torch.int, device=device
                )
            ) ** torch.arange(0, dtype_bitwidth, dtype=torch.int, device=device).flip(
                dims=(-1,)
            ).expand(
                (*injectee_shape, dtype_bitwidth)
            )

            filter = (
                p
                * nn.functional.dropout(
                    torch.ones_like(error_map, dtype=torch.float, device=device), 1 - p
                )
            ).int()

            error_map = (filter * error_map * scale_factor).sum(dim=-1).int()

        return error_map

    def error_inject(model, attack, sf, p):
        error_maps = {}

        for param_name, param in model.named_parameters():
            # Options for attacks are here, you can do weights in general bias in general
            # Then you can do specific kinds of weights/biases attn.weights, proj.weights
            if attack in param_name:  # or "bias" in param_name:
                injectee_shape = param.shape

                error_maps[param_name] = error_map(
                    injectee_shape, bitwidth, device, sf, p
                )

                error_fin = error_maps[param_name]

                param.data = (param.data.to(torch.int) ^ error_fin).to(torch.float)

    config = GPT2Config.from_pretrained("gpt2")

    config.gradient_checkpointing = True

    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

    state_dict = gpt2_model.state_dict()  # Get the model's state_dict
    # Create an instance of the modified model
    modified_model = GPT2LMHeadModel.from_pretrained(
        "gpt2", config=config, state_dict=state_dict
    )
    # Create a tokenizer for the model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Move the model to the specified device
    modified_model = modified_model.to(device)
    # Set the modified model to evaluation mode
    modified_model.eval()

    # Error injection needs a model which is gpt2 an attack name as a string sf as a scale factor to reduce errors and p to introduce randomness
    error_inject(modified_model, attack, sf, p)

    input_text = prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = modified_model.generate(
            input_ids, max_length=25, num_return_sequences=1
        )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
