def BartErrorInjector(input_text, num_params, new_val):
    def get_parameter_importance(model: nn.Module) -> dict:
        parameter_importance = {}
        for name, parameter in model.named_parameters():
            parameter_importance[name] = torch.std(
                parameter
            ).item()  
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

        selected_params = [param[0] for param in sorted_params[:num_params]]
        modified_params = set()

        for name, parameter in model.named_parameters():
            if name in selected_params:
                modified_params.add(name)
                modified_parameter = modification_func(parameter)
                parameter.data.copy_(modified_parameter)
            else:
                parameter.requires_grad_(False)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    modification_func = lambda parameter: parameter.clone().fill_(
        new_val
    ) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 

    modify_parameters(model, num_params, modification_func)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    summary_ids = model.generate(
        input_ids, num_beams=4, max_length=100, early_stopping=True
    )

    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    
    return str(summary)
