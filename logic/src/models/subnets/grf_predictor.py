import torch
import torch.nn as nn


class GatedRecurrentFillPredictor(nn.Module):
    def __init__(self, 
                input_dim=1,
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                activation='relu',
                af_param=1.0,
                threshold=6.0,
                replacement_value=6.0,
                n_params=3,
                uniform_range=[0.125, 1/3],
                bidirectional=False):
        super(GatedRecurrentFillPredictor, self).__init__()
        from ..modules import ActivationFunction
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Output dimension will be doubled if bidirectional
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            ActivationFunction(activation, af_param, threshold, replacement_value, n_params, uniform_range),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, fill_history):
        output, hidden = self.gru(fill_history)
        last_hidden = output[:, -1, :]
        predicted_fill = self.predictor(last_hidden)
        return predicted_fill

    def init_hidden(self, batch_size, device):
        # (num_layers * num_directions, batch_size, hidden_dim)
        directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * directions, batch_size, self.hidden_dim).to(device)
