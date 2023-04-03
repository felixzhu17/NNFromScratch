import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        self.hidden = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

X = datasets.load_boston()['data']
y = datasets.load_boston()['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

self = NeuralNetwork(input_size = X_train.shape[1], hidden_size = 50, output_size = 1)

optimizer = optim.SGD(self.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()
    output = self(torch.Tensor(X_train))
    loss = nn.MSELoss(output, torch.Tensor(y_train).view(-1,1))
    loss.backward()
    optimizer.step()
    print('Iteration:', i+1, 'Loss:', loss.item())

