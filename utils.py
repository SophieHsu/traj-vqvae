import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
from load_traj import TrajectoryDataset, MultiAgentTrajectoryDataset, traj_data_loaders, multiagent_traj_data_loaders
from human_knowledge_utils import BalancedAgentSampler, get_agent_to_indices
import numpy as np


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val

def data_loaders(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, **kwargs):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        # Updated: Use .data instead of .train_data
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    elif dataset == 'MINIGRID':
        # data_file = "data/minigrid/nored-lrf-mapupdate-penalty0.005.hdf5"
        data_file = "data/minigrid/4-rooms-small/6agents.hdf5" # 50k steps per agent version
        # data_file = "data/minigrid/4-rooms-1k/combined.hdf5" # 1k trajectories per agent version
        # full_dataset = TrajectoryDataset(data_file)
        full_dataset = MultiAgentTrajectoryDataset(data_file, kwargs["sequence_len"])
        n_total = len(full_dataset)
        n_val = int(n_total * 0.2)  # 20% for validation
        n_train = n_total - n_val
        print(f"Total trajectories: {n_total}, Training: {n_train}, Validation: {n_val}")

        # Split the dataset into training and validation subsets
        training_data, validation_data = random_split(full_dataset, [n_train, n_val])

        # Create DataLoaders for both subsets
        if kwargs["balanced_sampling"]:
            train_agent_to_data_idx = get_agent_to_indices(training_data)
            valid_agent_to_data_idx = get_agent_to_indices(validation_data)
            train_sampler = BalancedAgentSampler(train_agent_to_data_idx)
            valid_sampler = BalancedAgentSampler(valid_agent_to_data_idx)
            training_loader, validation_loader = multiagent_traj_data_loaders(
                training_data, validation_data, batch_size, 
                train_sampler=train_sampler, valid_sampler=valid_sampler,
            )
        else:  
            # training_loader, validation_loader = traj_data_loaders(training_data, validation_data, batch_size)
            training_loader, validation_loader = multiagent_traj_data_loaders(training_data, validation_data, batch_size)
        
        # Compute the variance of the training states (using the "state0" data)
        all_states = []
        for traj in training_data:
            all_states.append(traj["state0"])
        if all_states:
            all_states = np.concatenate(all_states, axis=0)
            x_train_var = np.var(all_states)
        else:
            x_train_var = 0.0
    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')

