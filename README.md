# Error Injection Process 

This repository contains information and resources related to the error injection process. The error injection process is a technique used to intentionally introduce errors or faults into a system for the purpose of evaluating its resilience, robustness, and error handling capabilities. 

## Table of Contents 

- [Introduction](#introduction) 
- [Getting Started](#getting-started) 
- [Getting Nodes](#Getting-Nodes) 
- [Error Injection Techniques](#error-injection-techniques) 
- [Graphing](#Graphing) 
- [Usage](#usage) 
- [Examples](#examples) 
- [Contributing](#contributing) 
- [License](#license) 

## Introduction 

In the field of Natural Language Processing (NLP), ensuring the robustness and reliability of models is crucial for their effective deployment in real-world applications. One key aspect of assessing the performance of NLP models is to evaluate their ability to handle errors and provide accurate results with errors injected into their nodes. More specifically we are looking to see how NLP Models handle hardware level bit errors. We are doing this by simulating errors with three novel error injection methods.  

## Getting Started 

To get started you will need **Placeholder** 

## Getting Nodes 

In order to fully understand what nodes make up a given NLP model we came up with [neuralNetPrint](important%20functions/neuralNetPrint.py/). This file contains two functions both of them doing two different things but they are very related.  

### allNodePrint  

allNodePrint prints out any given node of a neural network if it contains any word in the search parameter. This will give you a concatenated view of the tensors. It is recommended that you use general words for search such as “weight” or “bias” to find nodes to target. While you can search for a specific individual node it is highly recommended that you use nodePull instead. 

### nodePull  

nodePull is very similar to allNodePrint except that it is made for printing the entire tensor. The size of the print is solely limited by the size parameter. This function is useful to view a node before and after injection of errors to see changes. It works best in tandem with Manual Error Injection as both can be called on the same individual node (they both take it in as a parameter).  

## Error Injection Techniques 

All of our error injectors are doing their injection by attacking the individual nodes in a given NLP model. In our [important folders](important%20functions/) we have starter code for a few models that can help you get going.  

### Auto Error Injection 

Our first method of error injection is [Auto Error Injection](important%20functions/Auto%20Error%20Injectors/) which takes in the number of parameters you want changed and what you want them changed to. It operates by looking for the most important nodes and it then changes the value of the entire tensor to the value that you have given. There are significant limitations with this method one of them is over-injection where the model will almost always spit out gibberish due to the most important nodes being changed. Another important limitation is the lack of variety of outputs, due to the node that is chosen being changed to the same value no matter how often you run it, it will give back the same result.   

**Introduce node printing and tell the user that they can print nodes to verify error injection**  

### Manual Error Injection 

The next method of error injection is [Manual Error Injection](important%20functions/Manual%20Error%20Injectors/) which takes in a model, the node that you want to attack, scale factor which affects the error injection rate, and dropout rate which affects randomness and introduces another way to prevent over injection. This attack works by creating an error mask which takes the size of the tensor of the node that is being attacked, generates a tensor of that size that contains powers of two, uses dropout and scalefactor to introduce randomness, and then injects the errors through XOR.  

The benefits of this injector is primarily with its targeting capabilities; you are able to choose individual nodes that make up the model and attack them. This lets us isolate nodes and once injected will tell us how significant they really are. We have not explored the full capabilities of testing this with every node of a model and it may lead to a deeper understanding of how NLPs operate if done.  

### Bit Flip Injection 

The final method of error injection is [Bit Flip Injector](important%20functions/BitInjectors/) which takes in the index of the node you want changed (from the total list of nods in the model and how many you want changed. It operates by converting all elements of the tensor specified by the user to binary then flipping the number of bits specified by the user and converting all elements back to decimal. This new tensor replaces the old tensor in a modified model.  

The main benefit of this injector is its subtlety. When slightly increasing the number of bits to flip, the output of the model becomes more and more distorted. This is especially helpful in finding the breaking point of the model, where grammar and spelling mistakes become complete gibberish. Another benefit is that you can specifically choose which tensor to alter, allowing the user to see the impact each tensor has.  

## Graphing 

Use [Optimized Weight Distribution](WeightDistribution/optimizedDistributionFunction.py/) which pulls weights or biases. Use pull = ‘weight’ to graph weight distribution, pull = ‘bias’ for bias distribution. If graphing both weights and biases, change line 13 to “if ‘weigh’ or ‘bias’ in name:” and remove pull as a parameter of the function. To graph the distribution of a non-error injected model, simply import the model using the transformers library, and call the plot_weight_distribution function with the model and the desired pull as parameters. If graphing a modified model simply pass the modified_model as the model parameter instead.

Currently it uses bounds of -10 to 10 but this can be changed by editing the Start and Stop on line 37 and 38.

## Usage 

### Import your own models
1. Install dependencies from requirements.txt
2. Clone our repo
4. Choose a model to attack from HuggingFace's transformers library
6. Import and create an instance of the model to get an unaltered output
7. Use any of our error injectors prior to output generation
8. Graph and or use nodePull to see the difference before and after

### Starter Code
1. Install dependencies from requirements.txt
2. Clone our repo
3. Choose any starter code, Manual Error Injection requires you to print the function call to get an output, the others do not.
4. Inject errors as you see fit to see misgenerated outputs!
   
## Examples 

### Manual Error Injection

This can be used as a benchmark to see which nodes are more or less resilient against bit flip errors. 
For Example:
Here is our non error injected output with the input "Alex Fink wears hats"
```python
Generated text: Alex Fink wears hats. He also wears a lot of other hats, including a hat on his head. It's just one of many hats he's worn over the years. Click through the gallery to see more photos of him in a variety of hats and other items of clothing.
```
Then I imported the function as rF and I am attacking the model.encoder.layers.11.final_layer_norm.weight of the BART model.
```python
rF.error_inject(model, "model.encoder.layers.11.final_layer_norm.weight", .2, .05)
```

This is our misgenerated output.
```python
#Generated text: Alex Fink wears hats. He also has his own Twitter account, which he uses to post photos of himself in a variety of hats and sunglasses. In the last few weeks, he has been in the news for a number of reasons. Among them is his work for CNN.com, where he is a writer.
```
Here is a node that is more resistant or is less useful in generation.
```python
#rF.error_inject(model, "model.decoder.layers.11.self_attn.k_proj.weight", .2, .05)
```

We know this because the output is generally normal.
```python
#Generated text: Alex Fink wears hats. He also has his own Twitter account, which he uses to post photos of himself in a variety of hats and sunglasses. In the last few weeks, he has been in the news for a number of reasons. Among them is his work for CNN.com, where he is a writer.
```

Here is the before graph of the BART model, the weights and biases are being graphed. 
![BartGitPhoto](https://github.com/liambaker10/SHEAA/assets/123198090/ecd5783e-4af2-4e8f-8be6-a49a9210218d)





Here is the post error injetion <br>     
![BartGitPost](https://github.com/liambaker10/SHEAA/assets/123198090/7d838c7b-9d6f-4173-ba38-a5032c42794d)





## Contributing 

We welcome contributions from the community! If you find any issues or have suggestions for improvement, please fork our project! 

## License 

You are free to use, modify, and distribute this project in any way you see fit. 

## Acknowledgements 

We'd like to thank our mentor Dr. Xun Jiao from the ECE Department at Villanova University along with his PH.D student Ruixuan Wang who made this project possible.  

Thank you to L3Harris and Villanova University for funding this project. 

## Contact 

For any further contact please feel free to email any of us! 

sbains1@villanova.edu 
kseven@villanova.edu 
afink@villanova.edu 
lbaker10@vilanova.edu 

Note that we are undergraduate students, but we are motivated to learn as much as we can! 

Mentor:
xun.jiao@villanova.edu
--- 

  

 

 
