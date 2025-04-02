#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def collate_fn(batch):
    """
    Custom collate function to concatenate trajectories into fixed-length sequences
    """
    target_length = 30
    state0_batch = []
    state1_batch = []
    action_indices_batch = []
    reward_batch = []
    
    current_state0 = []
    current_state1 = []
    current_actions = []
    current_rewards = []
    
    for traj in batch:
        if len(current_state0) == 0:
            current_state0 = traj['state0']
            current_state1 = traj['state1']
            current_actions = traj['action_indices']
            current_rewards = traj['reward']
        else:
            current_state0 = np.concatenate([current_state0, traj['state0']])
            current_state1 = np.concatenate([current_state1, traj['state1']])
            current_actions = np.concatenate([current_actions, traj['action_indices']])
            current_rewards = np.concatenate([current_rewards, traj['reward']])
        
        # When we have enough timesteps, add to batch
        if len(current_state0) >= target_length:
            state0_batch.append(np.array(current_state0[:target_length]))
            state1_batch.append(np.array(current_state1[:target_length]))
            action_indices_batch.append(np.array(current_actions[:target_length]))
            reward_batch.append(np.array(current_rewards[:target_length]))
            
            # Remove used timesteps
            current_state0 = []
            current_state1 = []
            current_actions = []
            current_rewards = []
            
    
    # Convert to tensors
    return {
        'state0': torch.tensor(np.stack(state0_batch), dtype=torch.float32),
        'state1': torch.tensor(np.stack(state1_batch), dtype=torch.float32),
        'action_indices': torch.tensor(np.stack(action_indices_batch), dtype=torch.long),
        'reward': torch.tensor(np.stack(reward_batch), dtype=torch.float32),
        'lengths': torch.tensor([target_length] * len(state0_batch))
    }

class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for trajectories stored in an HDF5 file.
    Each trajectory is stored as a sample with states, actions, rewards, and additional attributes.
    """
    def __init__(self, data_file_path: str):
        self.trajectories = []
        self.action_to_idx = {}  # Dictionary to store action name to index mapping
        self.idx_to_action = {}  # Dictionary to store index to action name mapping
        
        # Define action mapping
        action_mapping = {
            'forward': 0,
            'left': 1,
            'right': 2,
            'pickup': 3,
            'drop': 4,
            'toggle': 5,
            'done': 6
        }
        
        with h5py.File(data_file_path, "r") as data_file:
            traj_group = data_file["trajectories"]
            for traj_name in traj_group.keys():
                grp = traj_group[traj_name]
                # Read datasets stored in each trajectory group
                state0 = np.array(grp["state0"])
                state1 = np.array(grp["state1"])
                action_names = np.array(grp["action_name"], dtype=str)
                reward = np.array(grp["reward"])
                
                # Convert action names to indices using the predefined mapping
                action_indices = np.array([action_mapping.get(name, -1) for name in action_names], dtype=np.int64)
                
                # Read any attributes stored with the trajectory
                attributes = {attr: grp.attrs[attr] for attr in grp.attrs.keys()}
                self.trajectories.append({
                    "state0": state0,
                    "state1": state1,
                    "action_indices": action_indices,
                    "action_names": action_names,
                    "reward": reward,
                    "attributes": attributes,
                })
        
        print("Action mapping:")
        for action, idx in action_mapping.items():
            print(f"  {idx}: {action}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def traj_data_loaders(train_data, val_data, batch_size):
    """
    Creates PyTorch DataLoaders for the training and validation datasets.
    """
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(
        description="Load trajectories from an HDF5 file, split into train/val sets, and create DataLoaders."
    )
    parser.add_argument("--data_file", type=str, required=False, default='data/minigrid/nored-lrf-mapupdate-penalty0.005.hdf5',
                        help="Path to the HDF5 data file containing all trajectories")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for data loaders")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of trajectories to use for validation (between 0 and 1)")
    args = parser.parse_args()

    # Create dataset object for all trajectories
    dataset = TrajectoryDataset(args.data_file)
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    print(f"Total trajectories: {n_total}, Training: {n_train}, Validation: {n_val}")

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Create DataLoaders from the training and validation datasets
    train_loader, val_loader = traj_data_loaders(train_dataset, val_dataset, args.batch_size)

    # Example: iterate over one batch from the training DataLoader
    for batch in train_loader:
        print("A batch from train_loader:")
        print("Batch shapes:")
        print(f"state0: {batch['state0'].shape}")
        print(f"state1: {batch['state1'].shape}")
        print(f"action_indices: {batch['action_indices'].shape}")
        print(f"reward: {batch['reward'].shape}")
        print(f"lengths: {batch['lengths']}")
        break

if __name__ == "__main__":
    main()
