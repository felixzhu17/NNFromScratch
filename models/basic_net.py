import torch
import torch.nn as nn
import numpy as np


class BasicPerceptronScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicPerceptronScratch, self).__init__()
        self.Wx_hidden = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.bias_hidden = nn.Parameter(torch.randn(hidden_size, 1))
        self.Wh_output = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (output_size + hidden_size)))
        )
        self.bias_output = nn.Parameter(torch.zeros(output_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.sigmoid(self.Wx_hidden @ x.T + self.bias_hidden)
        output = self.Wh_output @ hidden + self.bias_output
        return output.view(-1)


class BasicPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicPerceptron, self).__init__()
        self.linear_hidden = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.linear_output = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.linear_hidden.weight)
        nn.init.xavier_uniform_(self.linear_output.weight)

    def forward(self, x):
        hidden = self.sigmoid(self.bn(self.linear_hidden(x)))
        hidden = self.dropout(hidden)
        output = self.linear_output(hidden)
        return output.view(-1)

class ResNetBasicPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResNetBasicPerceptron, self).__init__()
        self.linear_hidden_1 = nn.Linear(input_size, hidden_size)
        self.linear_hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_hidden_3 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_1 = self.sigmoid(self.linear_hidden_1(x))
        hidden_2 = self.sigmoid(self.linear_hidden_2(hidden_1))
        hidden_3 = self.sigmoid(self.linear_hidden_3(hidden_2) + hidden_1)
        output = self.linear_output(hidden_3)
        return output.view(-1)


# RAW IMPLEMENTATION


class NumpyNeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.01,
        momentum=0.9,
        dropout_rate=0.05,
        lambda_l2=0.01,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.lambda_l2 = lambda_l2

        self.weights1 = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(
            2 / (self.input_size + self.hidden_size)
        )
        self.biases1 = np.random.randn(self.hidden_size, 1)
        self.weights2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(
            2 / (self.hidden_size + self.output_size)
        )
        self.biases2 = np.random.randn(self.output_size, 1)

        self.prev_weights1_update = np.zeros_like(self.weights1)
        self.prev_biases1_update = np.zeros_like(self.biases1)
        self.prev_weights2_update = np.zeros_like(self.weights2)
        self.prev_biases2_update = np.zeros_like(self.biases2)

        self.prev_error = np.inf

    def forward(self, X, training=True):
        self.z1 = self.weights1 @ X.T + self.biases1
        self.a1 = self.sigmoid(self.z1)
        if training:
            self.dropout_mask1 = np.random.rand(*self.a1.shape) > self.dropout_rate
            self.a1 *= self.dropout_mask1
        self.z2 = self.weights2 @ self.a1 + self.biases2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def mse_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        self.error_delta = self.mse_derivative(y.T, self.a2)
        self.z2_delta = self.sigmoid_derivative(self.a2) * self.error_delta
        self.z1_delta = self.sigmoid_derivative(self.a1) * (
            self.weights2.T @ self.z2_delta
        )

        # Apply dropout mask during backpropagation
        self.z1_delta *= self.dropout_mask1

        self.weights2_delta = self.z2_delta @ self.a1.T / m
        self.biases2_delta = np.sum(self.z2_delta, axis=1, keepdims=True) / m

        self.weights1_delta = self.z1_delta @ X / m
        self.biases1_delta = np.sum(self.z1_delta, axis=1, keepdims=True) / m

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def print_mse(self, X, y):
        print(f"MSE: {self.mse(y.T, self.forward(X, training=False))}")

    def update_learning_rate(self, current_error):
        if current_error < self.prev_error:
            self.learning_rate *= 1.1
        else:
            self.learning_rate *= 0.5
        self.prev_error = current_error

    def update(self, y):
        current_error = self.mse(self.a2, y)
        self.update_learning_rate(current_error)

        weights1_update = (
            self.learning_rate * (self.weights1_delta + self.lambda_l2 * self.weights1)
            + self.momentum * self.prev_weights1_update
        )
        biases1_update = (
            self.learning_rate * self.biases1_delta
            + self.momentum * self.prev_biases1_update
        )
        weights2_update = (
            self.learning_rate * (self.weights2_delta + self.lambda_l2 * self.weights2)
            + self.momentum * self.prev_weights2_update
        )
        biases2_update = (
            self.learning_rate * self.biases2_delta
            + self.momentum * self.prev_biases2_update
        )

        self.weights1 -= weights1_update
        self.biases1 -= biases1_update
        self.weights2 -= weights2_update
        self.biases2 -= biases2_update

        self.prev_weights1_update = weights1_update
        self.prev_biases1_update = biases1_update
        self.prev_weights2_update = weights2_update
        self.prev_biases2_update = biases2_update

    def batch_gradient_descent(self, X, y):
        self.forward(X)
        self.backward(X, y)
        self.update(y)
        self.print_mse(X, y)

    def mini_batch_gradient_descent(self, X, y, batch_size):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            X_batch = X[start:end]
            y_batch = y[start:end]

            y_pred = self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.update(y_batch)

        self.print_mse(X, y)

    def stochastic_gradient_descent(self, X, y):
        self.mini_batch_gradient_descent(X, y, 1)


class NumpyNeuralNetworkQuickProp:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights1 = np.random.randn(self.hidden_size, self.input_size)
        self.biases1 = np.random.randn(self.hidden_size, 1)
        self.weights2 = np.random.randn(self.output_size, self.hidden_size)
        self.biases2 = np.random.randn(self.output_size, 1)

        self.prev_weights1_update = np.zeros_like(self.weights1)
        self.prev_biases1_update = np.zeros_like(self.biases1)
        self.prev_weights2_update = np.zeros_like(self.weights2)
        self.prev_biases2_update = np.zeros_like(self.biases2)

        self.weights1_delta = np.zeros_like(self.weights1)
        self.biases1_delta = np.zeros_like(self.biases1)
        self.weights2_delta = np.zeros_like(self.weights2)
        self.biases2_delta = np.zeros_like(self.biases2)

        self.prev_weights1_delta = np.zeros_like(self.weights1)
        self.prev_biases1_delta = np.zeros_like(self.biases1)
        self.prev_weights2_delta = np.zeros_like(self.weights2)
        self.prev_biases2_delta = np.zeros_like(self.biases2)
        self.prev_error = np.inf

    def forward(self, X):
        self.z1 = self.weights1 @ X.T + self.biases1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.weights2 @ self.a1 + self.biases2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def backward(self, X, y):

        m = X.shape[0]
        y = y.reshape(-1, 1)

        self.error_delta = self.mse_derivative(y.T, self.a2)
        self.z2_delta = self.sigmoid_derivative(self.a2) * self.error_delta
        self.z1_delta = self.sigmoid_derivative(self.a1) * (
            self.weights2.T @ self.z2_delta
        )

        self.weights2_delta = self.z2_delta @ self.a1.T / m
        self.biases2_delta = np.sum(self.z2_delta, axis=1, keepdims=True) / m
        self.weights1_delta = self.z1_delta @ X / m
        self.biases1_delta = np.sum(self.z1_delta, axis=1, keepdims=True) / m

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def print_mse(self, X, y):
        print(f"MSE: {self.mse(y.T, self.forward(X))}")

    def update(self):
        weights1_update = (
            self.weights1_delta
            * self.prev_weights1_update
            / (self.prev_weights1_delta - self.weights1_delta)
        )
        biases1_update = (
            self.biases1_delta
            * self.prev_biases1_update
            / (self.prev_biases1_delta - self.biases1_delta)
        )
        weights2_update = (
            self.weights2_delta
            * self.prev_weights2_update
            / (self.prev_weights2_delta - self.weights2_delta)
        )
        biases2_update = (
            self.biases2_delta
            * self.prev_biases2_update
            / (self.prev_biases2_delta - self.biases2_delta)
        )

        self.weights1 -= weights1_update
        self.biases1 -= biases1_update
        self.weights2 -= weights2_update
        self.biases2 -= biases2_update

        self.prev_weights1_update = weights1_update
        self.prev_biases1_update = biases1_update
        self.prev_weights2_update = weights2_update
        self.prev_biases2_update = biases2_update

        self.prev_weights2_delta = self.weights2_delta
        self.prev_biases2_delta = self.biases2_delta
        self.prev_weights1_delta = self.weights1_delta
        self.prev_biases1_delta = self.biases1_delta

    def batch_gradient_descent(self, X, y):
        self.forward(X)
        self.backward(X, y)
        self.update()
        self.print_mse(X, y)

    def mini_batch_gradient_descent(self, X, y, batch_size):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            X_batch = X[start:end]
            y_batch = y[start:end]

            y_pred = self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.update()

        self.print_mse(X, y)

    def stochastic_gradient_descent(self, X, y):
        self.mini_batch_gradient_descent(X, y, 1)
