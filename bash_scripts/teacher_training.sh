#!/bin/bash

python train_teacher.py \
--exp_name "9R_masked_mean_classifier" \
--teacher_model_type "MaskedMeanClassifier" \
--model_dir "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-color-30p-10f-balanced" \
--checkpoint_file "checkpoint_epoch_4999.pt" \
--raw_data_path "/home/ayanoh/traj-vqvae/data/minigrid/9-rooms-2.5k/6agents.hdf5" \
--vae_type "RNNVQVAE" \
--num_epochs 5000 \
--track

python train_teacher.py \
--exp_name "9R_final_step_classifier" \
--teacher_model_type "FinalStepClassifier" \
--model_dir "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-color-30p-10f-balanced" \
--checkpoint_file "checkpoint_epoch_1999.pt" \
--raw_data_path "/home/ayanoh/traj-vqvae/data/minigrid/9-rooms-2.5k/6agents.hdf5" \
--num_epochs 5000 \
--track

python train_teacher.py \
--exp_name "9R_final_step_balanced" \
--teacher_model_type "FinalStepClassifier" \
--data_path "/home/ayanoh/traj-vqvae/data/teacher/minigrid/9-rooms-2.5k/9R2.5k-color-30p-10f-balanced/6agents.hdf5" \
--num_epochs 5000 \
--track

python train_teacher.py \
--exp_name "9R_masked_mean_balanced" \
--teacher_model_type "MaskedMeanClassifier" \
--data_path "/home/ayanoh/traj-vqvae/data/teacher/minigrid/9-rooms-2.5k/9R2.5k-color-30p-10f-balanced/6agents.hdf5" \
--num_epochs 5000 \
--track

python train_teacher.py \
--exp_name "4R_masked_mean_balanced" \
--teacher_model_type "MaskedMeanClassifier" \
--data_path "/home/ayanoh/traj-vqvae/data/teacher/minigrid/4-rooms-2.5k-color/4R2.5k-color-30p-10f-balanced/6agents.h5py" \
--num_epochs 5000 \
--track

# python train_teacher.py \
# --exp_name "masked_mean_classifier" \
# --teacher_model_type "MaskedMeanClassifier" \
# --model_dir "/home/ayanoh/traj-vqvae/trained_vqvae_models/4R-30past-10future-4R2.5kcolor" \
# --checkpoint_file "checkpoint_epoch_999.pt" \
# --raw_data_path "/home/ayanoh/traj-vqvae/data/minigrid/4-rooms-2.5k-color/6agents.hdf5" \
# --num_epochs 5000 \
# --track