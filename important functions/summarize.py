from transformers import AutoModel, AutoConfig
import torchinfo
import torch


# Get the a summary of the model
def get_model_summary(model_name_or_path, input_shape=(1, 10)):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, config=config)

    try:
        model_info = model.__repr__()
        input_tensor = torch.randint(0, model.config.vocab_size, input_shape)
        print(input_tensor)
        torchinfo.summary(model, input_data=input_tensor)
    except Exception as e:
        print("Failed to retrieve model summary.")
        print(f"Error: {str(e)}")


get_model_summary("gpt2")
