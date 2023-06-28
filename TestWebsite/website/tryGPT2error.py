from flask import Flask, Blueprint, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
import torch
import torch.nn as nn

tryGPT2error = Blueprint("tryGPT2error", __name__)


@tryGPT2error.route("/tryGPT2error")
def tryModel():
    return render_template("tryGPT2error.html")


@tryGPT2error.route("/tryGPT2error/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    number_parameters = request.form["num_parameters"]
    new_value = request.form["new_value"]
    num_params = int(number_parameters)
    input = msg
    new_val = float(new_value)

    modified_text = GPT2ErrorInjector(
        input, num_params=num_params, new_val=new_val
    )  # Inject errors into input text

    # generated_text = get_chat_response(
    #     modified_text, max_len
    # )  # Generate response with error-injected GPT-2
    return str(modified_text)

    # return get_chat_response(modified_text, max_len)


# def get_chat_response(text, max_length=50):
#     model_path = "modified_gpt2"
#     tokenizer_path = "modified_gpt2_tokenizer"

#     model = GPT2LMHeadModel.from_pretrained(model_path)
#     tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

#     generator = pipeline("text-generation", model="gpt2")

#     output = generator(text, max_length=max_length, num_return_sequences=1)

#     formatted = output[0]["generated_text"]
#     return formatted


def GPT2ErrorInjector(input_text, num_params, new_val):
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

        selected_params = [param[0] for param in sorted_params]
        modified_params = set()

        for name, parameter in model.named_parameters():
            if len(modified_params) < num_params and name in selected_params:
                modified_params.add(name)
                modified_parameter = modification_func(parameter)
                parameter.data.copy_(modified_parameter)
            else:
                parameter.requires_grad_(False)

    # Create the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Specify the number of distinct parameters to modify and the modification function
    modification_func = lambda parameter: parameter.clone().fill_(
        new_val
    )  # Example modification: set the parameter values to 0.0

    # Modify a specific number of distinct parameters of highest importance
    modify_parameters(model, num_params, modification_func)

    # Set the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate text using the modified model
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
