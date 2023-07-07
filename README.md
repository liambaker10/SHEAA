# Error Injection Process

This repository contains information and resources related to the error injection process. The error injection process is a technique used to intentionally introduce errors or faults into a system for the purpose of evaluating its resilience, robustness, and error handling capabilities.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Error Injection Techniques](#error-injection-techniques)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the field of Natural Language Processing (NLP), ensuring the robustness and reliability of models is crucial for their effective deployment in real-world applications. One key aspect of assessing the performance of NLP models is to evaluate their ability to handle errors and provide accurate results with errors injected into their nodes. More specifically we are lookig to see how NLP Models handle hardware level bit errors. We are doing this by simulating errors with three novel error injection methods. 

## Getting Started

To get started you will need **Placeholder**

## Error Injection Techniques

All of our error injectors are doing their injection by attacking the indivdual nodes in a given NLP model. In our [SHEAA/important functions] folders we have starter code for a few models that can help you get going. 

### Auto Erorr Injection

Our first method of error injection is [Auto Error Injection](SHEAA/important functions/Auto Error Injectors) which takes in the number of paramters you want changed and what you want them changed to. It operates by looking for the most important nodes and it then changes the value of the entire tensor to the value that you have given. There are signficant limitaitons with this method one of them is over-injection where the model will almost always spit out gibberish due to the most important nodes being changed. Another important limitation is the lack of variety of outputs, due to the node that is chosen being changed to the same value no matter how often you run it it will give back the same result.  **Alex check**

**Introduce node printing and tell the user that they can print nodes to verify error injection** 

### Manual Error Injection

The next method of error injection is [Manual Error Injection](SHEAA/important funcitons/Manual Error Injectors) which takes in a model, the node that you want to attack, scale factor which affects the error injection rate, and dropout rate which affects randomness and introduces another way to prevent over injeciton. This attack works by creating an error mask which takes the size of the tensor of the node that is being attacked, generates a tensor of that size that contains powers of two, uses dropout and scalefactor to introduce randomness, and then injects the errors through XOR. 

The benefits of this injector is primarily with its targeting capabilties; you are able to choose indivdual nodes that make up the model and attack them. This lets us isolate nodes and once injected will tell us how signficant they really are. We have not explored the full capabilities of testing this with every node of a model and it may lead to a deeper understanding of how NLPs operate if done. 

### Bit Flip Injection

The final method of error injection is [Bit Flip Injector](SHEAA/important functions/BitInjectors) which takes in the index of the node you want changed (from the total list of nods in the model and how many you want changed. **Alex explain**


## Usage

Provide detailed instructions on how to use the error injection process in practice. Include information about the tools, libraries, or frameworks that are required. Explain the steps involved in setting up the error injection environment, configuring error scenarios, and executing the error injection process.

## Examples

In this section, provide examples of how the error injection process can be applied in real-world scenarios. Include sample code, configurations, or test cases that demonstrate the effectiveness of error injection for improving system resilience.

## Contributing

We welcome contributions from the community! If you find any issues or have suggestions for improvement, please fork our project!

## License

You are free to use, modify, and distribute this project in any way you see fit.

## Acknowledgements

We'd like to thank our mentor Dr. Xun Jiao from the ECE Department at Villanova University along with his PH.D student Ruixuan Wang who made this project possible. 

Thank you to L3Harris and Villanova Univerity for funding this project.

## Contact

For any further contact please feel free to email any of us!

sbains1@villanova.edu
kseven@villanova.edu
afink@villanova.edu
lbaker10@vilanova.edu

Note that we are undergradute students but we are motivated to learn as much as we can!

---


