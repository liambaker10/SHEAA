import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim

#from z3 import *

class TestNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(TestNetwork, self).__init__()
        self.linear = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = None)

    def forward(self, x):
        g = self.linear(x)
        g = self.relu(g)
        g = self.linear2(g)
        g = self.relu(g)
        g = self.out(g)
        g = self.softmax(g)
        return g
    
def gen_rnd_data(truth_table_size, num_training_samples):
    #Get matrix of num_training_samples rows by truth_table_size columns
    rnd_data = np.random.randint(0, 2, size =(num_training_samples, truth_table_size))
    #Switch the 0s to be -1s
    rnd_data = np.where(rnd_data == 0, -1, rnd_data)
    #Convert to a tensor
    return torch.tensor(rnd_data, dtype=torch.float32)

def gen_labels(training_data_inputs):
    labels = []
    for row in training_data_inputs:
        if int(row[0]) == -1:
            labels.append(0)
        elif int(row[0]) == 1:
            labels.append(1)
    
    labels = torch.tensor(labels, dtype=torch.long)
    return F.one_hot(labels, num_classes = 2).float()

inputs = gen_rnd_data(16, 1000)
labels = gen_labels(inputs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = TestNetwork(16, 2).to(device)
#Loss function
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_values = []
test_loss_values = []

num_iters = 0
max_iters = 100

input_data = gen_rnd_data(16, 100)
label_data = gen_labels(input_data)

### VALIDATION ###
val_inputs = gen_rnd_data(16, 100)
val_labels = gen_labels(val_inputs)

### TESTING ###
test_inputs = gen_rnd_data(16, 100)
test_labels = gen_labels(test_inputs)

#Dummy data point for analysis
inputs, labels = (input_data, label_data)

while num_iters < max_iters:
    # zero the parameter gradients
    optimizer.zero_grad()
    #print("INPUTS:")
    #print(inputs)
    outputs = model(inputs)
    #print("OUTPUTS:")
    #print(outputs)
    #print("ORG WEIGHTS")
    #for p in list(model.parameters()):
    #    print(p.data)
    #Calculate loss
    loss = criterion(outputs, labels)

    #Test Outputs
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_labels)
    test_loss_values.append(test_loss.item())
    #print("LOSS 1")
    #print(loss.data)

    #Perform Backward pass
    loss.backward()
    #print("Gradients")
    #print(model.linear.weight.grad)

    #Update Model
    optimizer.step()
    #print("NEXT WEIGHTS")
    #for p in list(model.parameters()):
    #    print(p.data)
    
    print("LOSS:")      
    print(loss.item())
    #Append loss after each iteration
    loss_values.append(loss.item())

    num_iters += 1

plt.plot(loss_values, label="train")
plt.plot(test_loss_values, label="test")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

### TESTING ###
with torch.no_grad():
    val_outputs = model(val_inputs)
print("Output 0 comparison of network")
print(val_outputs[0])
print(test_labels[0])