#!/bin/bash

python minigrid_main_v2.py \
--n_embeddings 512 \
--num_epochs 5000 \
--embedding_loss_weight 1.0 \
--prediction_loss_weight 1.0 \
--save_interval 500 \
--input_seq_len 30 \
--in_dim 263 \
--state_dim 261 \
--data_file "/home/ayanoh/traj-vqvae/data/minigrid/9-rooms-2.5k/6agents.hdf5" \
--plot_dir 9R-past30-future10-9R2.5kcolor-balanced \
--track