#!/bin/bash

# # if generating data
# python train_teacher.py \
# --exp_name "masked_mean_classifier" \
# --teacher_model_type "MaskedMeanClassifier" \
# --model_dir "/home/ayanoh/traj-vqvae/trained_vqvae_models/4R-30past-10future-4R2.5kcolor" \
# --checkpoint_file "checkpoint_epoch_999.pt" \
# --raw_data_path "/home/ayanoh/traj-vqvae/data/minigrid/4-rooms-2.5k-color/6agents.h5py" \
# --num_epochs 5000 \
# # --track


# if using existing data
python train_teacher.py \
--exp_name "masked_mean_classifier" \
--teacher_model_type "MaskedMeanClassifier" \
--data_path "/home/ayanoh/traj-vqvae/data/teacher/test.hdf5" \
--num_epochs 5000 \
# --track