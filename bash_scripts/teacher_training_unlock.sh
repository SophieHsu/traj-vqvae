#!/bin/bash

python train_teacher_unlock.py \
--exp_name "batched-no-humanness-check-no-belief-supervision" \
--save_interval 500 \
--hidden_dim 256 \
--lstm_size 128 \
--num_epochs 5000 \
--num_envs 32 \
--n_rollout_steps_for_voi_computation 4 \
--num_steps 128 \
--student_model_dir "/home/ayanoh/human-knowledge/proxy_models/9-rooms" \
--vqvae_model_dir "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-binary-30p-10f-balanced" \
--vqvae_checkpoint_file "checkpoint_epoch_1999.pt" \
--track

