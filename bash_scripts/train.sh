#!/bin/bash


### VQVAE ###
python minigrid_main_v2.py \
--n_embeddings 512 \
--num_epochs 1000 \
--embedding_loss_weight 1.0 \
--prediction_loss_weight 1.0 \
--num_epochs 1000 \
--save_interval 200 \
--n_future_steps 10 \
--plot_dir debug \
--track

### Gumbel (Codebook-Free) ###
python minigrid_train_gumbel_vae.py \
--exp_name 9R_codefree_gumbel_6embs \
--n_embeddings 6 \
--num_epochs 1000 \
--in_dim 263 \
--state_dim 261 \
--embedding_loss_weight 1.0 \
--prediction_loss_weight 1.0 \
--num_epochs 1000 \
--save_interval 200 \
--n_future_steps 10 \
--plot_dir debug \
--track