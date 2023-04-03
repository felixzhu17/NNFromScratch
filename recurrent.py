import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.Wx = np.random.randn(hidden_size, input_size)
        self.Wh = np.random.randn(hidden_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev):
        h = np.tanh(self.Wx @ x + self.Wh @ h_prev + self.bh)
        y = self.sigmoid(self.Wy @ h + self.by)
        return y, h
    
# Example usage
input_size, hidden_size, output_size = 1, 5, 1
rnn = RecurrentNeuralNetwork(input_size, hidden_size, output_size)

x = np.array([[0.5]])
h_prev = np.zeros((hidden_size, 1))
y_target = np.array([[0.6]])

# Forward step
y, h = rnn.forward(x, h_prev)
print("Output:", y)
print("New hidden state:", h)
