import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from models.encoder import Encoder, RNNEncoder, SegmentRNNEncoder
from models.quantizer import VectorQuantizer, MultiCodebookVectorQuantizer
from models.decoder import Decoder, TransformerActionPredictor, MultiCodeTransformerActionPredictor, TransformerFutureActionPredictor

class PredictionTypes(Enum):
    """
    Define which timesteps the model predicts
    """
    FUTURE = 0 # predict future steps only
    PAST_FUTURE = 1 # predict past and future steps

"""Predict only the future n actions"""
class RNNFutureVQVAE(nn.Module):
    def __init__(
        self, in_dim, state_dim,
        h_dim, 
        n_embeddings, bidirectional, 
        beta,
        n_actions, n_steps,
        decoder_context_dim,
        n_attention_heads,
        n_decoder_layers,
        save_img_embedding_map=False,
    ):
        super().__init__()
        self.prediction_type = PredictionTypes.FUTURE

        self.n_actions = n_actions
        # encode image into continuous latent space
        self.encoder = RNNEncoder(in_dim=in_dim, h_dim=h_dim, num_layers=1, bidirectional=True)
        # pass continuous latent vector through discretization bottleneck
        embedding_dim = h_dim * 2 if bidirectional else h_dim
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, encoder_type="timernn")

        # decode the discrete latent representation
        self.decoder = TransformerFutureActionPredictor(
            embedding_dim=embedding_dim, # dimension of z_q
            state_dim=state_dim, # input state dimension (s, a, r)
            num_actions=n_actions,
            n_steps=n_steps,
            d_model=decoder_context_dim,
            nhead=n_attention_heads,
            num_layers=n_decoder_layers,
            # n_past_steps=30,
        )

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "future_traj" : [B, T_max, d_feature] # rest of trajectory after "traj"
            "mask" : [B, T],
            "future_mask: :
        }
        """
        traj = x["traj"]
        next_state = x["next_state"]
        future_actions = x["future_actions"]
        mask = x["mask"]
        future_mask = x["future_mask"]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj)
        # breakpoint()
        # last_real_index = mask.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), axis=1)
        # z_e = z_e[torch.arange(z_e.shape[0]), last_real_index]
        # pass through VQ layer
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e, mask)
        
        # predict future
        target_actions = future_actions[:,:self.decoder.n_steps]
        logits = self.decoder(z_q, next_state, target_actions)

        # compute CE loss, masking out effects from timesteps where ground truth prediction target does not exist
        future_mask = future_mask[:,:self.decoder.n_steps]
        masked_targets = target_actions.clone() # masking padded timesteps
        masked_targets[future_mask == 0] = -100  # index to ignore
        
        # check for case where all target actions are masked out
        if (masked_targets != -100).sum() == 0:
            pred_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            pred_loss = F.cross_entropy(
                logits.view(-1, self.n_actions),
                masked_targets.reshape(-1),
                ignore_index=-100,
            )
        if np.isnan(pred_loss.item()):
            print("prediction loss is nan. do masked_targets and logits look okay?")
            breakpoint()

        return logits, embedding_loss, pred_loss
    
    def get_embeddings(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "mask" : [B, T],
        }
        """
        traj = x["traj"]
        mask = x["mask"]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj)

        # pass through VQ layer
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e, mask)
        return z_e, z_q, min_encodings, min_encoding_indices

"""Predict the past actions and future n actions"""
class RNNVQVAE(nn.Module):
    def __init__(
        self, in_dim, state_dim,
        h_dim, 
        n_embeddings, bidirectional, 
        beta,
        n_actions, n_past_steps, n_future_steps,
        decoder_context_dim,
        n_attention_heads,
        n_decoder_layers,
        save_img_embedding_map=False,
    ):
        super().__init__()
        self.prediction_type = PredictionTypes.PAST_FUTURE
        self.n_past_steps = n_past_steps
        self.n_future_steps = n_future_steps
        
        self.n_actions = n_actions
        # encode image into continuous latent space
        self.encoder = RNNEncoder(in_dim=in_dim, h_dim=h_dim, num_layers=1, bidirectional=True)
        # pass continuous latent vector through discretization bottleneck
        embedding_dim = h_dim * 2 if bidirectional else h_dim
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, encoder_type="timernn")

        # decode the discrete latent representation
        self.decoder = TransformerActionPredictor(
            embedding_dim=embedding_dim, # dimension of z_q
            state_dim=state_dim, # input state dimension (s, a, r)
            num_actions=n_actions,
            n_future_steps=n_future_steps,
            n_past_steps=n_past_steps,
            d_model=decoder_context_dim,
            nhead=n_attention_heads,
            num_layers=n_decoder_layers,
        )

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "future_traj" : [B, T_max, d_feature] # rest of trajectory after "traj"
            "mask" : [B, T],
            "future_mask: :
        }
        """
        traj = x["traj"]
        next_state = x["next_state"]
        actions = x["actions"]
        future_actions = x["future_actions"]
        mask = x["mask"]
        future_mask = x["future_mask"]

        # get timestep-wise continuous embedding
        z_e = self.encoder(traj, mask)
        # # get embedding corresponding to the last non-masked timestep for each trajectory
        # last_real_index = mask.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), axis=1)
        # z_e = z_e[torch.arange(z_e.shape[0]), last_real_index]
        # pass through VQ layer
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e, mask)

        # predict past and future
        # target_actions = future_actions[:,:self.decoder.n_steps]
        target_actions = torch.cat((actions, future_actions[:,:self.decoder.n_future_steps]), dim=1)
        logits = self.decoder(z_q, next_state, target_actions)

        # compute CE loss, masking out effects from timesteps where ground truth prediction target does not exist
        target_mask = torch.cat([mask, future_mask[:,:self.decoder.n_future_steps]], dim=1)
        masked_targets = target_actions.clone() # masking padded timesteps
        masked_targets[target_mask == 0] = -100  # index to ignore
        
        # check for case where all target actions are masked out
        if (masked_targets != -100).sum() == 0:
            pred_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            pred_loss = F.cross_entropy(
                logits.reshape(-1, self.n_actions),
                masked_targets.reshape(-1),
                ignore_index=-100,
            )
        if np.isnan(pred_loss.item()):
            print("prediction loss is nan. do masked_targets and logits look okay?")
            breakpoint()

        return logits, embedding_loss, pred_loss, min_encodings, min_encoding_indices
    
    def get_embeddings(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "mask" : [B, T],
        }
        """
        traj = x["traj"]
        mask = x["mask"]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj, mask)

        # pass through VQ layer
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e, mask)
        return z_e, z_q, min_encodings, min_encoding_indices


class RNNVQVAEK(nn.Module):
    def __init__(
        self, in_dim, state_dim,
        h_dim, 
        n_embeddings, bidirectional, 
        beta,
        n_actions, n_steps,
        decoder_context_dim,
        n_attention_heads,
        n_decoder_layers,
        k=4,
        save_img_embedding_map=False,
    ):
        super().__init__()
        self.n_actions = n_actions

        self.encoder = SegmentRNNEncoder(in_dim=in_dim, h_dim=h_dim, num_layers=1, bidirectional=True, k=k)
        embedding_dim = self.encoder.out_dim
        self.vector_quantization = MultiCodebookVectorQuantizer(n_embeddings, embedding_dim, beta)

        self.decoder = MultiCodeTransformerActionPredictor(
            embedding_dim, 
            state_dim, 
            num_actions=n_actions, 
            n_steps=n_steps, 
            d_model=decoder_context_dim, 
            nhead=n_attention_heads, 
            num_layers=n_decoder_layers
        )

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "future_traj" : [B, T_max, d_feature] # rest of trajectory after "traj"
            "mask" : [B, T],
            "future_mask: :
        }
        """
        traj = x["traj"]
        next_state = x["next_state"]
        target_actions = x["future_actions"][:,:self.decoder.n_steps]
        mask = x["mask"]
        future_mask = x["future_mask"][:,:self.decoder.n_steps]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj, mask)

        # pass through VQ layer
        embedding_loss, z_q, perplexity, indices = self.vector_quantization(z_e)
        
        # predict future
        logits = self.decoder(z_q, next_state, target_actions)

        # compute CE loss, masking out effects from timesteps where ground truth prediction target does not exist
        masked_targets = target_actions.clone() # masking padded timesteps
        masked_targets[future_mask == 0] = -100  # index to ignore

        # Masked cross-entropy loss
        pred_loss = F.cross_entropy(
            logits.view(-1, self.n_actions),
            target_actions.reshape(-1),
            reduction='none'
        )
        mask_flat = future_mask.reshape(-1).float()
        pred_loss = (pred_loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)

        return logits, embedding_loss, pred_loss
    
    def get_embeddings(self, x):
        """
        x = {
            "traj" : [B, T, d_feature], # train data
            "mask" : [B, T],
        }
        """
        traj = x["traj"]
        mask = x["mask"]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj, mask)

        # pass through VQ layer
        embedding_loss, z_q, perplexity, min_encoding_indices = self.vector_quantization(z_e)

        return z_e, z_q, min_encoding_indices
   


class VQVAE(nn.Module):
    def __init__(
        self, h_dim, res_h_dim, n_res_layers,
        n_embeddings, embedding_dim, beta, save_img_embedding_map=False,
        encoder_conv_kernel=4,
        encoder_conv_stride=2,
    ):
        super().__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim, stride=encoder_conv_stride, kernel=encoder_conv_kernel)
        self.pre_quantization_conv = nn.Conv1d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, encoder_type="conv")
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        # print("input x shape", x.shape)
        # print("reshaped x shape", x.reshape(x.shape[0], 1, -1).shape)
        z_e = self.encoder(x.reshape(x.shape[0], 1, -1))
        # print("encoder output shape", z_e.shape)
        z_e = self.pre_quantization_conv(z_e)
        # print("pre quantization conv output shape", z_e.shape)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        # print("zq shape", z_q.shape)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity