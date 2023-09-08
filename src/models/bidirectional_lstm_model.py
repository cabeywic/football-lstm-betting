import torch.nn as nn
import torch


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, output_dim)  # *2 because of bidirection

    def forward(self, x):
        # Get the device of the input tensor (x)
        device = x.device
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(device).requires_grad_()  # *2 because of bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(device).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out
