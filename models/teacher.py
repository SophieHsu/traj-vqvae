
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Teacher with belief head and VoI prediction critic
"""
class BeliefHead(nn.Module):
    """
    Inputs: z_q, previous teacher action
    Outputs: belief over student types
    """

    def __init__(
        self,
        z_dim,
        n_teacher_actions,
        n_student_types,
        action_embedding_dim=16,
        hidden_dim=128,       
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_embedding_dim = action_embedding_dim
        self.n_teacher_actions = n_teacher_actions
        self.n_student_types = n_student_types
        self.hidden_dim = hidden_dim

        self.teacher_action_embed_layer = nn.Embedding(n_teacher_actions, action_embedding_dim) # embed 1-hot teacher action 
        self.fc1 = nn.Linear(z_dim + action_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_student_types)

    def forward(self, zq, prev_teacher_action):
        """
        Args:
            zq : student trajectory embedding (with temporal dimension flattened)
            prev_teacher_action : index of teacher action from previous step
        """
        action_emb = self.teacher_action_embed_layer(prev_teacher_action)
        x = torch.cat([zq, action_emb], dim=-1)
        
        x = F.relu(self.fc1(x))
        belief_logits = self.fc2(x)

        belief_probs = F.softmax(belief_logits, dim=-1)

        return belief_logits, belief_probs

class VoICriticHead(nn.Module):
    """
    Inputs: belief over student types, current environment state
    Output: VoI predictions
    """
    def __init__(
        self,
        n_student_types,
        n_teacher_actions,
        state_dim,
        hidden_dim=128,
    ):
        super().__init__()
        self.n_student_types = n_student_types
        self.n_teacher_actions = n_teacher_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(n_student_types + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_teacher_actions)

    def forward(self, s_t, belief_prob):
        x = torch.cat([s_t, belief_prob], dim=-1)
        x = F.relu(self.fc1(x))
        voi_preds = self.fc2(x)

        return voi_preds

class VoICriticTeacherModel(nn.Module):
    """
    Teacher model with
    - belief head: keeps belief over student types
        - inputs: discrete student trajectory embedding (z_q), previous teacher action (a_T)_{t-1}
        - outputs: probability distribution over student types
    - critic that predicts VoI for each teacher action
        - inputs: output of belief head (b_t), current environment state (s_t)
    """
    def __init__(
        self, 
        z_dim,
        state_dim, 
        n_teacher_actions, 
        n_student_types,
        action_embedding_dim=16,
        hidden_dim=128,
    ):
        """
        Args:
            z_dim : dimension of z_q
            state_dim : dimension of environment state s_t
            n_teacher_actions : number of available teacher actions
            n_student_types : number of possible student types
            action_embedding_dim : dimension to embed teacher action into
            hidden_dim : size of hidden layer 
        """
        super().__init__()
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.n_teacher_actions = n_teacher_actions
        self.n_student_types = n_student_types
        self.hidden_dim = hidden_dim

        self.belief_head = BeliefHead(
            z_dim=z_dim,
            action_embedding_dim=action_embedding_dim,
            n_teacher_actions=n_teacher_actions,
            n_student_types=n_student_types,
            hidden_dim=hidden_dim,
        )
        self.voi_critic_head = VoICriticHead(
            n_student_types=n_student_types,
            n_teacher_actions=n_teacher_actions,
            state_dim=state_dim,
            hidden_dim=hidden_dim, 
        )

    def forward(self, zq, prev_teacher_action, s_t):
        
        belief_logits, belief_probs = self.belief_head(zq, prev_teacher_action)
        voi_preds = self.voi_critic_head(s_t, belief_probs)

        return {
            "belief_logits": belief_logits,
            "belief_probs": belief_probs,
            "voi_preds": voi_preds,
        }
        


"""

"""

class BeliefModel(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim, dropout=0.1):
        """
        Args:
            embedding_dim (int): Dimension of the z_q embeddings (D).
            num_classes (int): Number of target classes.
            hidden_dim (int, optional): If provided, adds a hidden layer before classification.
            dropout (float): Dropout probability (only used if hidden_dim is set).
        """
        super().__init__()
        self.use_hidden = hidden_dim is not None
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if self.use_hidden:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_classes)
            )
        else:
            self.classifier = nn.Linear(self.in_dim, self.num_classes)
    
    def forward(self, x):
        logits = self.classifier(x)
        return logits

    # def forward(self, z_q, s_t, mask):
    #     """
    #     Args:
    #         z_q (tensor): quantized embeddings of shape (B, T, D)
    #         s_t (tensor): most recent ground truth environment state
    #         mask (tensor): binary mask of shape (B, T), where 1 = valid, 0 = padded

    #     Returns:
    #         logits (tensor): output logits of shape (B, num_classes)
    #     """
    #     # mask and take temporal mean of z_q
    #     masked_z_q = z_q * mask.unsqueeze(-1)  # (B, T, D)
    #     sum_z_q = masked_z_q.sum(dim=1)  # (B, D)
    #     valid_counts = mask.sum(dim=1, keepdim=True)  # (B, 1)
    #     mean_z_q = sum_z_q / (valid_counts + 1e-6)  # (B, D)

    #     # concatenate state 
    #     x = torch.cat([mean_z_q, s_t], dim=-1)

    #     # compute logits
    #     logits = self.classifier(x)

    #     return logits
 

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
