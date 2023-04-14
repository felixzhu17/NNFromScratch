import numpy as np
import torch
import torch.nn as nn


class RecurrentNeuralNetworkScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNeuralNetworkScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.Wy = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.by = nn.Parameter(torch.zeros(output_size, 1))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        h_hist = [torch.zeros(self.hidden_size, 1)]
        for t in range(len(x)):
            x_t = x[t].view(-1, 1)
            h_t_1 = h_hist[-1]
            h_t = self.tanh(self.Wx @ x_t + self.Wh @ h_t_1 + self.bh)
            h_hist.append(h_t)
        h_hist = h_hist[1:]
        y_pred = [self.Wy @ h_t + self.by for h_t in h_hist]
        return torch.cat(y_pred).view(-1)


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        x = x.view(
            1, x.size(0), self.input_size
        )  # We only ever do one batch since we are training one step ahead
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        y_pred = self.fc(out)
        return y_pred.view(-1)


class JordanNetworkScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(JordanNetworkScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_size, output_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.Wy = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.by = nn.Parameter(torch.zeros(output_size, 1))

        self.learning_rate = learning_rate
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        y_t_1 = torch.zeros(self.output_size, 1)
        y_pred = []

        for t in range(len(x)):
            x_t = x[t].view(-1, 1)
            h_t = self.tanh(self.Wx @ x_t + self.Wh @ y_t_1 + self.bh)
            y_t = self.Wy @ h_t + self.by
            y_pred.append(y_t)
            y_t_1 = y_t

        return torch.cat(y_pred).view(-1)


class BiDirectionalRecurrentNetworkScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(BiDirectionalRecurrentNetworkScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx_forward = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_forward = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.bh_forward = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_backward = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_backward = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.bh_backward = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wy_forward = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.Wy_backward = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.by = nn.Parameter(torch.zeros(output_size, 1))

        self.learning_rate = learning_rate
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        self.h_forward_hist = [torch.zeros(self.hidden_size, 1)]
        for t in range(len(x)):
            x_t = x[t].view(-1, 1)
            h_forward_t_1 = self.h_forward_hist[-1]
            h_forward_t = self.tanh(
                self.Wx_forward @ x_t
                + self.Wh_forward @ h_forward_t_1
                + self.bh_forward
            )
            self.h_forward_hist.append(h_forward_t)
        self.h_forward_hist = self.h_forward_hist[1:]

        self.h_backward_hist = [torch.zeros(self.hidden_size, 1)]
        for t in reversed(range(len(x))):
            x_t = x[t].view(-1, 1)
            h_backward_t_1 = self.h_backward_hist[-1]
            h_forward_t = self.tanh(
                self.Wx_backward @ x[t].view(-1, 1)
                + self.Wh_backward @ h_backward_t_1
                + self.bh_backward
            )
            self.h_backward_hist.append(h_forward_t)
        self.h_backward_hist = self.h_backward_hist[1:][::-1]

        self.y_pred = [
                self.Wy_forward @ h_forward_t
                + self.Wy_backward @ h_backward_t
                + self.by
            for h_forward_t, h_backward_t in zip(
                self.h_forward_hist, self.h_backward_hist
            )
        ]
        return torch.cat(self.y_pred).view(-1)


class BiDirectionalRecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(BiDirectionalRecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # Bi-directional requires two sets of weights
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        x = x.view(
            1, x.size(0), self.input_size
        )  # We only ever do one batch since we are training one step ahead
        h0 = torch.zeros(
            2, x.size(0), self.hidden_size
        )  # Bi-directional requires two sets of weights
        out, _ = self.rnn(x, h0)
        y_pred = self.fc(out)
        return y_pred.view(-1)


class GatedRecurrentUnitScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(GatedRecurrentUnitScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx_reset = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_reset = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.b_reset = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_candidate = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_candidate = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.b_candidate = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_update = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_update = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.bh_update = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wy = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.by = nn.Parameter(torch.zeros(output_size, 1))

        self.learning_rate = learning_rate
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        self.h_hist = [torch.zeros(self.hidden_size, 1)]
        for t in range(len(x)):
            x_t = x[t].view(-1, 1)
            h_t_1 = self.h_hist[t]

            r_t = self.sigmoid(
                self.Wx_reset @ x_t + self.Wh_reset @ h_t_1 + self.b_reset
            )
            candidate_t = self.tanh(
                self.Wx_candidate @ x_t
                + self.Wh_candidate @ (r_t * h_t_1)
                + self.b_candidate
            )
            z_t = self.sigmoid(
                self.Wx_update @ x_t + self.Wh_update @ h_t_1 + self.bh_update
            )
            h_t = z_t * h_t_1 * (1 - z_t) * candidate_t
            self.h_hist.append(h_t)
        self.h_hist = self.h_hist[1:]
        self.y_pred = [self.Wy @ h_t + self.by for h_t in self.h_hist]
        return torch.cat(self.y_pred).view(-1)

class LSTMScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super(LSTMScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx_forget = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_forget = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.b_forget = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_input = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_input = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.b_input = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_candidate = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_candidate = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.b_candidate = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wx_update = nn.Parameter(
            torch.randn(hidden_size, input_size)
            * torch.sqrt(torch.tensor(2.0 / (input_size + hidden_size)))
        )
        self.Wh_update = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
            * torch.sqrt(torch.tensor(1.0 / hidden_size))
        )
        self.bh_update = nn.Parameter(torch.zeros(hidden_size, 1))

        self.Wy = nn.Parameter(
            torch.randn(output_size, hidden_size)
            * torch.sqrt(torch.tensor(2.0 / (hidden_size + output_size)))
        )
        self.by = nn.Parameter(torch.zeros(output_size, 1))

        self.learning_rate = learning_rate
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        self.y_hist = []
        self.h_hist = [torch.zeros(self.hidden_size, 1)]
        self.c_hist = [torch.zeros(self.hidden_size, 1)]

        for t in range(len(x)):
            x_t = x[t].view(-1, 1)
            h_t_1 = self.h_hist[t]
            c_t_1 = self.c_hist[t]

            f_t = self.sigmoid(
                self.Wx_forget @ x_t + self.Wh_forget @ h_t_1 + self.b_forget
            )

            i_t = self.sigmoid(
                self.Wx_input @ x_t + self.Wh_input @ h_t_1 + self.b_input
            )

            candidate_t = self.tanh(
                self.Wx_candidate @ x_t
                + self.Wh_candidate @ h_t_1
                + self.b_candidate
            )

            c_t = f_t * c_t_1 + i_t * candidate_t

            o_t = self.sigmoid(
                self.Wx_update @ x_t + self.Wh_update @ h_t_1 + self.bh_update
            )
            h_t = o_t * c_t
            y_t = self.Wy @ h_t + self.by

            self.h_hist.append(h_t)
            self.c_hist.append(c_t)
            self.y_hist.append(y_t)

        self.h_hist = self.h_hist[1:]
        self.y_pred = [self.Wy @ h_t + self.by for h_t in self.h_hist]
        return torch.cat(self.y_pred).view(-1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(1, x.size(0), self.input_size)
        h0 = torch.zeros(
            1, x.size(0), self.hidden_size
        ) 
        c0 = torch.zeros(
            1, x.size(0), self.hidden_size
        ) 
        out, _ = self.lstm(x, (h0, c0))
        y_pred = self.linear(out)
        return y_pred.view(-1)

class AttentionScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        self.attention_softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.view(1, x.size(0), self.input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        hidden, _ = self.lstm(x, (h0, c0))
        hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])

        # Attention
        key = self.key(hidden)
        query = self.query(hidden)
        value = self.value(hidden)

        attention = key @ query.T
        weighting = self.attention_softmax(attention)
        context = weighting @ value
        y_pred = self.output(context)
        return y_pred.view(-1)


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(1, x.size(0), self.input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        hidden, _ = self.lstm(x, (h0, c0))

        # Attention
        attn_output, _ = self.attention(hidden, hidden, hidden)
        y_pred = self.output(attn_output)
        return y_pred.view(-1)


# RAW IMPLEMENTATION


class NumpyRecurrentNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wx = np.random.randn(hidden_size, input_size) * np.sqrt(
            2 / (input_size + hidden_size)
        )
        self.Wh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * np.sqrt(
            2 / (hidden_size + output_size)
        )
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        h_prev = np.zeros((self.hidden_size, 1))
        self.y_pred, self.h_hist = [], [h_prev]
        for t in range(len(x)):
            h_t = np.tanh(self.Wx @ x[t].reshape(-1, 1) + self.Wh @ h_prev + self.bh)
            y_t = self.Wy @ h_t + self.by
            self.y_pred.append(y_t)
            self.h_hist.append(h_t)
            h_prev = h_t
        return self.y_pred

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def print_mse(self, X, y):
        print(f"MSE: {self.mse(y.T, np.concatenate(self.forward(X)).flatten())}")

    def backward(self, x, y):
        Wy_delta_list = []
        by_delta_list = []
        Wh_delta_list = []
        Wx_delta_list = []
        bh_delta_list = []

        h_future_delta = np.zeros_like(np.zeros((self.hidden_size, 1)))

        for t in reversed(range(len(x))):
            y_delta = self.mse_derivative(
                y[t], self.y_pred[t]
            )
            h_current_delta = (self.Wy.T @ y_delta) + h_future_delta
            h_raw_delta = h_current_delta * self.tanh_derivative(self.h_hist[t])
            h_future_delta = self.Wh.T @ h_raw_delta

            Wh_delta = h_raw_delta @ self.h_hist[t - 1].T
            Wx_delta = h_raw_delta @ x[t].reshape(-1, 1)
            Wy_delta = y_delta @ self.h_hist[t].T
            bh_delta = h_raw_delta
            by_delta = y_delta

            Wy_delta_list.append(Wy_delta)
            by_delta_list.append(by_delta)
            Wh_delta_list.append(Wh_delta)
            Wx_delta_list.append(Wx_delta)
            bh_delta_list.append(bh_delta)

        self.Wy -= np.sum(Wy_delta_list, axis=0) * self.learning_rate
        self.by -= np.sum(by_delta_list, axis=0) * self.learning_rate
        self.Wh -= np.sum(Wh_delta_list, axis=0) * self.learning_rate
        self.Wx -= np.sum(Wx_delta_list, axis=0) * self.learning_rate
        self.bh -= np.sum(bh_delta_list, axis=0) * self.learning_rate
