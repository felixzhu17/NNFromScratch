import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_length):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_length = output_length

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.decoder_lstm = nn.LSTM(input_size, hidden_size, batch_first = False)
        self.decoder_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, y_true = None, teacher_training = False):
        h_start, c_start = self.encoder(x)
        y_pred = self.decoder(h_start, c_start, y_true, teacher_training)
        return torch.cat(y_pred, dim=0).squeeze().T

    def encoder(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        _, (h_start, c_start) = self.encoder_lstm(x, (h0, c0))
        return h_start, c_start

    def decoder(self, h_start, c_start, y_true = None, teacher_training = False):
        h = h_start
        c = c_start
        y_pred = [self.decoder_output(h)]
        for i in range(self.output_length - 1):
            if teacher_training:
                input = y_true[:, i].view(1, -1, 1)
            else:
                input = y_pred[-1]
            _, (h, c) = self.decoder_lstm(input, (h, c))
            y = self.decoder_output(h)
            y_pred.append(y)
        return y_pred

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_length, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_length = output_length

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.decoder_self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.encoder_decoder_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, y_true = None, teacher_training = False):
        encoder_output = self.encoder(x)
        y_pred = self.decoder(x, encoder_output, y_true, teacher_training)
        return y_pred.squeeze()

    def encoder(self, x):
        x = x.view(x.size(0), x.size(1), 1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        hidden, _ = self.encoder_lstm(x, (h0, c0))
        encoder_output, _ = self.encoder_self_attention(hidden, hidden, hidden)
        return encoder_output

    def decoder(self, x, encoder_output, y_true = None, teacher_training = False):
        y = torch.zeros(x.size(0), 1, 1)
        y_pred = torch.empty(x.size(0), 0, 1)
        for i in range(self.output_length):
            h0 = torch.zeros(1, y.size(0), self.hidden_size)
            c0 = torch.zeros(1, y.size(0), self.hidden_size)
            hidden, _ = self.decoder_lstm(y, (h0, c0))
            decoder_output, _ = self.decoder_self_attention(hidden, hidden, hidden)
            decoder_output, _ = self.encoder_decoder_attention(decoder_output, encoder_output, encoder_output)
            prediction = self.decoder_linear(decoder_output[:,-1,:])
            if teacher_training:
                y = torch.cat((y, y_true[:, i].view(-1, 1, 1)), dim=1)
            else:
                y = torch.cat((y, prediction.unsqueeze(2)), dim=1)
            y_pred = torch.cat((y_pred, prediction.unsqueeze(2)), dim=1)

        return y_pred