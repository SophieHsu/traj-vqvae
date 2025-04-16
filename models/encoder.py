
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stride=2, kernel=4):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


class RNNEncoder(nn.Module):
    """
    RNN-based encoder. Accepts variable-length 1D sequences.

    Inputs:
    - in_dim : input feature dimension per timestep
    - h_dim : hidden dimension of the RNN
    - num_layers : number of RNN layers
    - bidirectional : whether to use a bidirectional RNN
    """

    def __init__(self, in_dim, h_dim, num_layers=1, bidirectional=True,):
        super(RNNEncoder, self).__init__()
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

    def forward(self, x, lengths=None):
        """
        x: Tensor of shape (B, T, in_dim)
        """
        out, _ = self.rnn(x)

        return out  # Shape: (B, T, h_dim * num_directions)

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 1, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
