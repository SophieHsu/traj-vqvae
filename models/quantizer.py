import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiCodebookVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):  # z: [B, k, D]
        B, k, D = z.shape
        z_flat = z.reshape(-1, D)  # [B*k, D]

        # Compute distances
        d = (z_flat ** 2).sum(dim=1, keepdim=True) + \
            (self.embedding.weight ** 2).sum(dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())  # [B*k, n_e]

        indices = torch.argmin(d, dim=1)  # [B*k]
        z_q = self.embedding(indices).view(B, k, D)  # [B, k, D]

        # Loss
        z_reshaped = z.view(B * k, D)
        z_q_reshaped = z_q.view(B * k, D)
        commitment_loss = self.beta * F.mse_loss(z_reshaped, z_q_reshaped.detach())
        codebook_loss = F.mse_loss(z_reshaped.detach(), z_q_reshaped)
        loss = commitment_loss + codebook_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Perplexity
        one_hot = F.one_hot(indices, self.n_e).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, z_q, perplexity, indices.view(B, k)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, encoder_type):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.encoder_type = encoder_type

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        if self.encoder_type == "conv":
            # this encoder type outputs z_e with dimension (B, e_dim, C)
            z = z.permute(0, 2, 1).contiguous() # (B, e_dim, C) -> (B, C, e_dim)
            z_flattened = z.reshape(-1, self.e_dim)
        elif self.encoder_type == "trajrnn":
            z_flattened = z
        elif self.encoder_type == "timernn":
            z_flattened = z.reshape(-1, self.e_dim)
        else:
            raise NotImplementedError("encoder_type other than conv and rnn are not supported")
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding (mask out padded timesteps)
        mask = mask.unsqueeze(-1)
        commitment_loss = torch.mean(((z_q.detach()-z) * mask) **2)
        codebook_loss = torch.mean(((z_q - z.detach()) * mask) ** 2)
        loss = commitment_loss + self.beta * codebook_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        if self.encoder_type == "conv":
            z_q = z_q.permute(0, 2, 1).contiguous()
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices



class TrajlevelVectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, encoder_type):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.encoder_type = encoder_type

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        if self.encoder_type == "conv":
            # this encoder type outputs z_e with dimension (B, e_dim, C)
            z = z.permute(0, 2, 1).contiguous() # (B, e_dim, C) -> (B, C, e_dim)
            z_flattened = z.view(-1, self.e_dim)
        elif self.encoder_type == "rnn":
            z_flattened = z
        else:
            raise NotImplementedError("encoder_type other than conv and rnn are not supported")
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        if self.encoder_type == "conv":
            z_q = z_q.permute(0, 2, 1).contiguous()
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices

