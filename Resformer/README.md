# NN-Error-Injection

This is a tool for simulating random bit errors in neural network weights and neurons. By using this tool, we can find how robust different neural network is and how different layers influence their robustness.

## Models Compatibility
Any neural network models using PyTorch(torch.nn) layer.


## Getting started
1. Install numpy,pytorch,torchvision and bitstring. 
2. Download the project
3. Change the setting in the weight-error-injection.py before running. Including: model path, model name, dataset path, dataset path, dataset normalized value, number of repetitions, error rate, filter name(option)

An example setting is below:
```Bash
dataset_path = '../imagenet-mini'

# These normalized value for ImageNet
normalized_value_mean = [0.485, 0.456, 0.406]
normalized_value_std = [0.229, 0.224, 0.225]

# model name and path
model_name = 'googlenet'
path = 'googlenet.pt'

# error injection layer name
filter_name_list = ['conv3.conv','fc.weight']

# number of repetitions
num_exp = 2

# Error rate is from 10^-10 to 10^-6 (10^-10, 10^-9...)
min_error = 10
max_error = 5
```
4. After changing all settings, you can run weight-error-injection.py, and the result will be saved in csv file.

### Ways to define error inject layers
You can put all the keywords of your layer names into the filter list.
Put the model path in the layer-name-read.py and get the layer parameter name.

For example. vit_16_b's layer name is below:

```Bash
class_token
conv_proj.weight
conv_proj.bias
encoder.pos_embedding
encoder.layers.encoder_layer_0.ln_1.weight
encoder.layers.encoder_layer_0.ln_1.bias
encoder.layers.encoder_layer_0.self_attention.in_proj_weight
encoder.layers.encoder_layer_0.self_attention.in_proj_bias
encoder.layers.encoder_layer_0.self_attention.out_proj.weight
encoder.layers.encoder_layer_0.self_attention.out_proj.bias
encoder.layers.encoder_layer_0.ln_2.weight
encoder.layers.encoder_layer_0.ln_2.bias
encoder.layers.encoder_layer_0.mlp.0.weight
encoder.layers.encoder_layer_0.mlp.0.bias
encoder.layers.encoder_layer_0.mlp.3.weight
encoder.layers.encoder_layer_0.mlp.3.bias

…

encoder.layers.encoder_layer_11.ln_1.weight
encoder.layers.encoder_layer_11.ln_1.bias
encoder.layers.encoder_layer_11.self_attention.in_proj_weight
encoder.layers.encoder_layer_11.self_attention.in_proj_bias
encoder.layers.encoder_layer_11.self_attention.out_proj.weight
encoder.layers.encoder_layer_11.self_attention.out_proj.bias
encoder.layers.encoder_layer_11.ln_2.weight
encoder.layers.encoder_layer_11.ln_2.bias
encoder.layers.encoder_layer_11.mlp.0.weight
encoder.layers.encoder_layer_11.mlp.0.bias
encoder.layers.encoder_layer_11.mlp.3.weight
encoder.layers.encoder_layer_11.mlp.3.bias
encoder.ln.weight
encoder.ln.bias
heads.head.weight
heads.head.bias
```

If you want to inject all mlp layers and and layer normalization:
```Bash
filter_name_list = [‘mlp’,’ln’]
```

If you want to inject error for encoder0 and encoder1:
```Bash
filter_name_list = [‘encoder_layer_0’,’encoder_layer_1’]
```

### Evaluation
Result save in model_name.csv and detail-model_name.csv files with error rate, top1 and top5 accuracy.

model_name.csv only record the average accuracy for each error rate.

detail-model_name.csv record every data point from the expreiment. 



### Fault-injection
Random bit flips for each value in the parameter. Current only support float32. More data type will add soon.

## Link to paper


## Citation
If you use ``, please cite us:

## Contact Us
For any further questions please contact <szhang6@villanova.edu>
