
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch.nn.functional as F
import numpy as np
import os
import sys

class AgentClassifier(nn.Module): # TODO - IP
    """
    Classifier that takes (state, action, reward) trajectories as input and predict the ID of the agent that generated the trajectory
    RNN-based encoding layers take variable length trajectories as input, and the embedding is passed to a prediction head

    Inputs:
    - in_dim : input feature dimension per timestep
    - h_dim : hidden dimension of the RNN
    - num_layers : number of RNN layers
    - bidirectional : whether to use a bidirectional RNN
    """

    def __init__(self, in_dim, h_dim, n_agents=6, num_layers=1, bidirectional=True,):
        super(AgentClassifier, self).__init__()
        self.h_dim = h_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.classifier = nn.Sequential(
            nn.Linear(h_dim * self.num_directions, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents)  # Predict 6 agent IDs
        )

    def forward(self, x, mask, lengths=None):
        """
        x: Tensor of shape (B, T, in_dim)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
        else:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)

        # Sort by lengths (required by pack_padded_sequence)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]

        packed_x = pack_padded_sequence(x_sorted, lengths_sorted, batch_first=True)
        packed_out, _ = self.rnn(packed_x)

        # pad the output back to the original shape
        out_sorted, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1]) 

        # Unsort to restore original batch order
        _, unsort_idx = sort_idx.sort()
        out = out_sorted[unsort_idx]

        return out
