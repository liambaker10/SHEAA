from flask import Flask, Blueprint, render_template, request
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForCausalLM

tryRoBERTaError = Blueprint("tryRoBERTaError", __name__)


@tryRoBERTaError.route("/tryRoBERTaError")
def tryModel():
    option_values = [
        "LayerNorm.weight",
        "word_embeddings.weight",
        "position_embeddings.weight",
        "token_type_embeddings.weight",
        "output.dense.weight",
        "LayerNorm.bias",
        "self.value.bias",
        "output.dense.bias",
    ]
    return render_template("tryRoBERTaError.html", option_values=option_values)


@tryRoBERTaError.route("/tryRoBERTaError/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    number_parameters = request.form["num_parameters"]
    new_value = request.form["new_value"]
    num_params = int(number_parameters)
    input = msg
    new_val = float(new_value)
    error_injection_type = request.form["category"]
    dropout_rate = float(request.form["dropout"])
    scale_factor = float(request.form["scale"])

    if error_injection_type == "Alex":
        modified_text = RobertaErrorInjector(
            input, num_params=num_params, new_val=new_val
        )
        return str(modified_text)
    elif error_injection_type != None:
        return RobertaResponse(
            input, attack=error_injection_type, sf=scale_factor, p=dropout_rate
        )
    else:
        return "did not work"


def RobertaErrorInjector(input_text, num_params, new_val):
    def get_parameter_importance(model: nn.Module) -> dict:
        parameter_importance = {}
        for name, parameter in model.named_parameters():
            parameter_importance[name] = torch.std(
                parameter
            ).item()  # Calculate importance based on standard deviation
        return parameter_importance

    def modify_parameters(
        model: nn.Module, num_params: int, modification_func: callable
    ):
        parameter_importance = get_parameter_importance(model)
        sorted_params = sorted(
            parameter_importance.items(), key=lambda x: x[1], reverse=True
        )
        total_params = len(sorted_params)

        if num_params > total_params:
            num_params = total_params

        selected_params = [param[0] for param in sorted_params][:num_params]
        modified_params = set()

        for name, parameter in model.named_parameters():
            if len(modified_params) < num_params and name in selected_params:
                modified_params.add(name)
                modified_parameter = modification_func(parameter)
                parameter.data.copy_(modified_parameter)
            else:
                parameter.requires_grad_(False)

    # Load the RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load the RoBERTa model
    model = RobertaForCausalLM.from_pretrained("roberta-base")

    # Specify the number of distinct parameters to modify and the modification function
    num_params = num_params  # Number of parameters to modify
    modification_func = lambda parameter: parameter.clone().fill_(
        new_val
    )  # Modification function

    # Modify a specific number of distinct parameters of highest importance
    modify_parameters(model, num_params, modification_func)

    # Set the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize the input text
    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Find the position of the masked token
    try:
        masked_index = token_ids.index(tokenizer.mask_token_id)
    except ValueError:
        # Handle case when mask token is not found
        return "Mask token not found in the input text"

    # Create a list to store the output sentences
    output_sentences = []

    # Generate multiple mask-filled sentences
    for _ in range(1):
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
    output_text = input_text.replace("<mask>", predicted_token)
    output_sentences.append(output_text)

    # Concatenate the sentences into a single paragraph with new lines
    output_paragraph = "\n".join(output_sentences)

    # Return the paragraph of output sentences
    return output_paragraph


def RobertaResponse(prompt, attack, sf=0.3, p=1e-4):
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

    model_name = "roberta-base"
    model = RobertaForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    input_text = prompt

    # Tokenize the input text
    input_tokens = tokenizer.encode_plus(
        input_text, add_special_tokens=True, padding="longest", return_tensors="pt"
    )

    input_tokens = input_tokens.to(device)

    # Error injection needs a model which is gpt2 an attack name as a string sf as a scale factor to reduce errors and p to introduce randomness
    error_inject(model, attack, sf, p)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            max_length=100,  # Specify the desired maximum length of the generated text
            num_beams=5,  # Specify the number of beams for beam search
            early_stopping=True,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
        )

    # Decode the generated output
    generated_text = tokenizer.decode(output.squeeze(), skip_special_tokens=True)

    return generated_text
    # Decode and print the generated text
