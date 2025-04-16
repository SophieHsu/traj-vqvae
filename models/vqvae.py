import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder, RNNEncoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


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
   
class RNNVQVAE(nn.Module):
    def __init__(
        self, in_dim, h_dim, 
        # num_layers, bidirectional,
        # res_h_dim, n_res_layers,
        n_embeddings, bidirectional, 
        beta, save_img_embedding_map=False,
    ):
        super().__init__()
        # encode image into continuous latent space
        self.encoder = RNNEncoder(in_dim=in_dim, h_dim=h_dim, num_layers=1, bidirectional=True)
        # self.pre_quantization_conv = nn.Conv1d(
        #     h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        embedding_dim = h_dim * 2 if bidirectional else h_dim
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, encoder_type="rnn")
        # # decode the discrete latent representation
        # self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        """
        x = {
            "traj" : [B, T, d_feature],
            "mask" : [B, T],
        }
        """
        traj = x["traj"]
        mask = x["mask"]

        # get embedding corresponding to the last non-masked timestep for each trajectory
        z_e = self.encoder(traj)
        last_real_index = mask.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), axis=1)
        z_e = z_e[torch.arange(z_e.shape[0]), last_real_index]

        # z_e = self.pre_quantization_conv(z_e)
        # print("pre quantization conv output shape", z_e.shape)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        breakpoint()
        # print("zq shape", z_q.shape)
        # x_hat = self.decoder(z_q)

        # if verbose:
        #     print('original data shape:', x.shape)
        #     print('encoded data shape:', z_e.shape)
        #     print('recon data shape:', x_hat.shape)
        #     assert False

        # return embedding_loss, x_hat, perplexity
        return z_e
   