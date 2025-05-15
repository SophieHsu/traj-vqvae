
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMeanClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim=None, dropout=0.1):
        """
        Args:
            embedding_dim (int): Dimension of the z_q embeddings (D).
            num_classes (int): Number of target classes.
            hidden_dim (int, optional): If provided, adds a hidden layer before classification.
            dropout (float): Dropout probability (only used if hidden_dim is set).
        """
        super().__init__()
        self.use_hidden = hidden_dim is not None

        if self.use_hidden:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, z_q, mask):
        """
        Args:
            z_q (Tensor): Quantized embeddings of shape (B, T, D)
            mask (Tensor): Binary mask of shape (B, T), where 1 = valid, 0 = padded

        Returns:
            logits (Tensor): Output logits of shape (B, num_classes)
        """
        masked_z_q = z_q * mask.unsqueeze(-1)  # (B, T, D)
        sum_z_q = masked_z_q.sum(dim=1)  # (B, D)
        valid_counts = mask.sum(dim=1, keepdim=True)  # (B, 1)
        mean_z_q = sum_z_q / (valid_counts + 1e-6)  # (B, D)
        logits = self.classifier(mean_z_q)
        return logits

class FinalStepClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim=None, dropout=0.1):
        """
        Uses the last unpadded timestep of the trajectory embedding as the input and predict which agent generated the trajectory
        Args:
            embedding_dim (int): Dimension of the z_q embeddings (D).
            num_classes (int): Number of target classes.
            hidden_dim (int, optional): If provided, adds a hidden layer before classification.
            dropout (float): Dropout probability (only used if hidden_dim is set).
        """
        super().__init__()
        self.use_hidden = hidden_dim is not None

        if self.use_hidden:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, z_q, mask):
        """
        Args:
            z_q (Tensor): Quantized embeddings of shape (B, T, D)
            mask (Tensor): Binary mask of shape (B, T), where 1 = valid, 0 = padded

        Returns:
            logits (Tensor): Output logits of shape (B, num_classes)
        """
        last_real_index = mask.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), axis=1)
        z_q_last = z_q[torch.arange(z_q.shape[0]), last_real_index]
        logits = self.classifier(z_q_last)
        return logits
