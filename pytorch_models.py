import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from keras import datasets
from torch.utils.data import TensorDataset
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

# x_train = transform(x_train)

# x_train, y_train = torch.from_numpy(x_train).type(torch.float), torch.from_numpy(y_train).type(torch.long)
x_train, y_train = torch.from_numpy(x_train).type(torch.float), torch.from_numpy(y_train).type(torch.long)

# x_train, y_train = torch.from_numpy(x_train).type(torch.float), torch.from_numpy(y_train).type(torch.long)
x_test, y_test = torch.from_numpy(x_test).type(torch.float), torch.from_numpy(y_test).type(torch.long)

print(x_train)


trainset = TensorDataset(x_train, y_train)
valset = TensorDataset(x_test, y_test)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)