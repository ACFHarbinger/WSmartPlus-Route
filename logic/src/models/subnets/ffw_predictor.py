import torch.nn as nn

from ..modules import ActivationFunction


class FeedForwardWeightPredictor(nn.Module):
    def __init__(self, input_dim, activation='relu', num_weights=4, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, num_weights),  
            nn.Softmax(dim=-1)
        )
    
    def forward(self, instance_features, keys):
        weights = self.network(instance_features)
        cost_dict = {k: weights[:, i] for i, k in enumerate(keys)}
        return cost_dict
