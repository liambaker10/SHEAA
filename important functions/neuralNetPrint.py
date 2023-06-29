#Prints all nodes of a given model that include anything in the search parameter
def allNodePrint(model, search):
    for name, param in model.named_parameters():
        if search in name:
            output = str(param)[:50] + '...' if len(str(param)) > 50 else str(param)
            # Print the parameter name, shape, and the truncated output
            print(f"Parameter Name: {name}")
            print(f"Parameter Shape: {param.shape}")
            print(f"Parameter Tensor: {output}")
            print()

#Prints individual nodes and their full tensor for whatever size you want. Could be used to print more than 1 node but not recommended 
def nodePull(model, search, size):
    for name, param in model.named_parameters():
        if search in name:
            output = param.flatten()[:size]
            # Print the parameter name, shape, and the truncated output
            print(f"Parameter Name: {name}")
            print(f"Parameter Shape: {param.shape}")
            print(f"Parameter Tensor: {output}")
            print()

