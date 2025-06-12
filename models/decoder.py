
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.residual import ResidualStack, LinearResidualStack

class MultiCodeTransformerActionPredictor(nn.Module):
    def __init__(
        self, embedding_dim, state_dim, num_actions, n_steps,
        d_model=128, nhead=4, num_layers=3
    ):
        super().__init__()
        self.n_steps = n_steps
        self.state_proj = nn.Linear(state_dim, d_model)
        self.code_proj = nn.Linear(embedding_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.token_embed = nn.Embedding(num_actions, d_model)
        self.action_out = nn.Linear(d_model, num_actions)

    def forward(self, z_q, current_state, target_actions=None):
        """
        z_q: [B, k, D] – VQ latents
        current_state: [B, state_dim]
        target_actions: [B, T_out] – optional for teacher forcing
        """
        B = z_q.shape[0]
        memory = self.code_proj(z_q)  # [B, k, d_model]

        state_embed = self.state_proj(current_state).unsqueeze(1)  # [B, 1, d_model]
        memory = torch.cat([state_embed, memory], dim=1)  # [B, k+1, d_model]

        if target_actions is not None:
            tgt = self.token_embed(target_actions)  # [B, T_out, d_model]
        else:
            tgt = torch.zeros(B, self.n_steps, memory.size(-1), device=memory.device)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_steps).to(memory.device)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # [B, T_out, d_model]
        return self.action_out(output)  # [B, T_out, num_actions]


class TrajLevelTransformerActionPredictor(nn.Module):
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


class TransformerFutureActionPredictor(nn.Module):
    """
    Predicts future n actions given z_q
    """
    def __init__(self, embedding_dim, state_dim, num_actions, n_steps,
                 d_model, nhead, num_layers):
        super().__init__()
        self.n_steps = n_steps
        self.num_actions = num_actions

        self.state_proj = nn.Linear(state_dim, d_model)
        self.token_embedding = nn.Embedding(num_actions, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(n_steps+1, d_model)) # +1 since starting with next-sate token

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers=num_layers
        )

        self.latent_proj = nn.Linear(embedding_dim, d_model)
        self.out = nn.Linear(d_model, num_actions)

    def forward(self, z_q, state, target_actions=None):
        """
        z_q: [B, T_latent, D]
        state: [B, state_dim]
        target_actions: [B, n_steps] (optional, used for teacher forcing)
        """
        B = z_q.shape[0]

        # encode memory from discrete latents
        memory = self.latent_proj(z_q).transpose(0, 1)  # [T_latent, B, d_model]

        # project state
        state_embed = self.state_proj(state).unsqueeze(1)  # [B, 1, d_model]

        if target_actions is not None:
            tokens = self.token_embedding(target_actions)  # [B, n_steps, d_model]
            tokens = torch.cat([state_embed, tokens[:, :-1]], dim=1)  # shift right
        else:
            tokens = state_embed  # inference: start with state

        tokens += self.pos_embedding[:tokens.size(1)].unsqueeze(0)  # add pos embedding
        tokens = tokens.transpose(0, 1)  # [n_steps, B, d_model]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_steps).to(tokens.device)

        out = self.transformer(tgt=tokens, memory=memory, tgt_mask=tgt_mask)
        logits = self.out(out.transpose(0, 1))  # [B, n_steps, num_actions]
        # logits = self.out(out.transpose(0, 1))[:, 1:]  # discard state part of sequence?

        return logits
    
class TransformerActionPredictor(nn.Module):
    """
    Predicts past (all timesteps used as input) and future n steps of actions given z_q
    """
    def __init__(
        self, embedding_dim, 
        state_dim, num_actions, n_future_steps, n_past_steps,
        d_model, nhead, num_layers
    ):
        """
        n_past_steps (int) : number of steps passed in as input to the encoder
        """
        super().__init__()
        self.n_future_steps = n_future_steps
        self.n_past_steps = n_past_steps
        self.num_actions = num_actions

        self.state_proj = nn.Linear(state_dim, d_model)

        self.pad_token_id = num_actions # special token for padded actions (assuming action indices are 0~num_actions-1)
        self.token_embedding = nn.Embedding(num_actions+1, d_model, padding_idx=self.pad_token_id)

        n_prediction_steps = self.n_future_steps + self.n_past_steps
        self.pos_embedding = nn.Parameter(torch.randn(n_prediction_steps + 1, d_model))  # +1 for state token

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers=num_layers
        )

        self.latent_proj = nn.Linear(embedding_dim, d_model)
        self.out = nn.Linear(d_model, num_actions)

    def forward(self, z_q, z_q_mask, state, target_actions=None):
        """
        z_q: [B, T_latent, D]
        state: [B, state_dim]
        target_actions: [B, n_steps] (optional, used for teacher forcing)
        """
        B = z_q.shape[0]

        # encode memory from discrete latents
        memory = self.latent_proj(z_q).transpose(0, 1)  # [T_latent, B, d_model]

        # project state
        state_embed = self.state_proj(state).unsqueeze(1)  # [B, 1, d_model]
        if target_actions is not None:
            # replace -100 with padding token ID so padded actions are not embedded
            target_actions = target_actions.clone()
            target_actions[target_actions == -100] = self.pad_token_id
            tgt_key_padding_mask = (target_actions == self.pad_token_id)  # Mask padding tokens

            tokens = self.token_embedding(target_actions)  # [B, T, d_model]
            tokens = torch.cat([state_embed, tokens[:, :-1]], dim=1)  # shift right
        else:
            tokens = state_embed
            tgt_key_padding_mask = torch.zeros(B, 1, dtype=torch.bool).to(tokens.device)  # No padding for input tokens

        tokens += self.pos_embedding[:tokens.size(1)].unsqueeze(0) # add pos embedding
        tokens = tokens.transpose(0, 1)  # [n_steps, B, d_model]

        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_steps).to(tokens.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(0)).to(tokens.device)
        memory_key_padding_mask = (z_q_mask == 0) # True where padded

        # out = self.transformer(tgt=tokens, memory=memory, tgt_mask=tgt_mask)
        out = self.transformer(tgt=tokens, memory=memory, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        logits = self.out(out.transpose(0, 1))  # [B, n_steps, num_actions]
        # logits = self.out(out.transpose(0, 1))[:, 1:]  # discard state part of sequence?

        return logits
    
    @torch.no_grad()
    def autoregressive_decode(self, z_q, state, n_past_steps, n_future_steps):
        """
        Autoregressively generate actions given z_q and current state.
        
        z_q: [B, T_latent, D]
        state: [B, state_dim]
        max_len: total number of steps to generate (n_past + n_future)

        Returns:
            predicted_actions: [B, max_len] (int tokens)
        """
        max_len = n_past_steps + n_future_steps
        B = z_q.shape[0]
        device = z_q.device

        # encode memory
        memory = self.latent_proj(z_q).transpose(0, 1)  # [T_latent, B, d_model]

        # fixed context
        state_embed = self.state_proj(state).unsqueeze(1)  # [B, 1, d_model]

        # initialize sequence with state_embed
        generated_tokens = []
        input_tokens = state_embed  # shape [B, 1, d_model]

        for step in range(max_len):
            # add positional embedding
            tokens = input_tokens + self.pos_embedding[:input_tokens.size(1)].unsqueeze(0)
            tokens = tokens.transpose(0, 1)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(0)).to(device)
            out = self.transformer(tgt=tokens, memory=memory, tgt_mask=tgt_mask)
            logits = self.out(out[-1])  # take only last token's output: shape [B, num_actions]

            predicted_action = torch.argmax(logits, dim=-1)  # [B]

            generated_tokens.append(predicted_action)

            # Embed and append predicted action
            next_token = self.token_embedding(predicted_action).unsqueeze(1)  # [B, 1, D]
            input_tokens = torch.cat([input_tokens, next_token], dim=1)

        return torch.stack(generated_tokens, dim=1)  # [B, max_len]



# class TransformerActionPredictor(nn.Module):
#     def __init__(self, embedding_dim, state_dim, num_actions, n_steps, d_model=128, nhead=4, num_layers=2):
#         """
#         Transformer to predict future agent actions from knowledge embedding and current state

#         Args:
#             embedding_dim : dimension of VQ layer output z_q
#             state_dim : size of state (per timestep) the action predictor sees (does not include action and reward components)
#             num_actions : number of possible discrete actions
#             n_steps : number of future steps to predict
#             d_model : input gets projected into a context of this size
#             nhead : number of attention heads in multi-head attention
#             num_layers :
#         """
#         super().__init__()
#         self.n_steps = n_steps
#         self.num_actions = num_actions

#         self.latent_project = nn.Linear(embedding_dim + state_dim, d_model) # project z_q and current state to context
#         self.action_embed = nn.Embedding(num_actions + 1, d_model)  # +1 for start token
#         self.pos_embed = nn.Parameter(torch.randn(n_steps, d_model)) # learnable positional embeddings

#         decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True) 
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers) 

#         self.output_proj = nn.Linear(d_model, num_actions) # output projection head

#     def forward(self, latent, state, target_actions=None):
#         """
#         latent: (B, D)
#         state: (B, S)
#         target_actions: (B, T) or None (for autoregressive inference)

#         Returns:
#             logits: (B, T, num_actions)
#         """
#         B = latent.size(0)
#         T = self.n_steps

#         # Encode conditioning context
#         context = torch.cat([latent, state], dim=-1)  # (B, D + S)
#         context = self.latent_project(context).unsqueeze(1)  # (B, 1, d_model)

#         # Prepare target action sequence with start token
#         if target_actions is not None:
#             tgt = torch.cat([
#                 torch.full((B, 1), self.num_actions, dtype=torch.long, device=latent.device),  # start token
#                 target_actions[:, :-1]  # shift targets right
#             ], dim=1)
#         else:
#             tgt = torch.full((B, T), self.num_actions, dtype=torch.long, device=latent.device)  # start tokens

#         tgt_emb = self.action_embed(tgt) + self.pos_embed.unsqueeze(0)  # (B, T, d_model)

#         # Generate causal mask
#         tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(latent.device)  # (T, T)

#         # Transformer decoder
#         output = self.transformer(tgt_emb, context, tgt_mask=tgt_mask)  # (B, T, d_model)

#         logits = self.output_proj(output)  # (B, T, num_actions)
#         return logits


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
