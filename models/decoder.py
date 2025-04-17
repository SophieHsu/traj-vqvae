
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.residual import ResidualStack, LinearResidualStack

class TransformerActionPredictor(nn.Module):
    def __init__(self, embedding_dim, state_dim, num_actions, n_steps, d_model=128, nhead=4, num_layers=2):
        """
        Transformer to predict future agent actions from knowledge embedding and current state

        Args:
            embedding_dim : dimension of VQ layer output z_q
            state_dim : size of state (per timestep) the action predictor sees (does not include action and reward components)
            num_actions : number of possible discrete actions
            n_steps : number of future steps to predict
            d_model : input gets projected into a context of this size
            nhead : number of attention heads in multi-head attention
            num_layers :
        """
        super().__init__()
        self.n_steps = n_steps
        self.num_actions = num_actions

        self.latent_project = nn.Linear(embedding_dim + state_dim, d_model) # project z_q and current state to context
        self.action_embed = nn.Embedding(num_actions + 1, d_model)  # +1 for start token
        self.pos_embed = nn.Parameter(torch.randn(n_steps, d_model)) # learnable positional embeddings

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True) 
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers) 

        self.output_proj = nn.Linear(d_model, num_actions) # output projection head

    def forward(self, latent, state, target_actions=None):
        """
        latent: (B, D)
        state: (B, S)
        target_actions: (B, T) or None (for autoregressive inference)

        Returns:
            logits: (B, T, num_actions)
        """
        B = latent.size(0)
        T = self.n_steps

        # Encode conditioning context
        context = torch.cat([latent, state], dim=-1)  # (B, D + S)
        context = self.latent_project(context).unsqueeze(1)  # (B, 1, d_model)

        # Prepare target action sequence with start token
        if target_actions is not None:
            tgt = torch.cat([
                torch.full((B, 1), self.num_actions, dtype=torch.long, device=latent.device),  # start token
                target_actions[:, :-1]  # shift targets right
            ], dim=1)
        else:
            tgt = torch.full((B, T), self.num_actions, dtype=torch.long, device=latent.device)  # start tokens

        tgt_emb = self.action_embed(tgt) + self.pos_embed.unsqueeze(0)  # (B, T, d_model)

        # Generate causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(latent.device)  # (T, T)

        # Transformer decoder
        output = self.transformer(tgt_emb, context, tgt_mask=tgt_mask)  # (B, T, d_model)

        logits = self.output_proj(output)  # (B, T, num_actions)
        return logits

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim//2, 1, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

class LinearDecoder(nn.Module):
    """
    Custom linear version of Decoder class
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(LinearDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, 1),
            LinearResidualStack(h_dim // 2, h_dim // 2, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.layers(x)



if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 1, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
