#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def trajectory_dataset_collate_fn(batch):
    """
    For use with TrajectoryDataset
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

def mutliagent_traj_collate_fn(batch):
    """
    Custom collate function for use with MultiAgentTrajectoryDataset
    """
    state0, state1, future_state = [], [], []
    action_indices, future_action_indices = [], []
    reward, future_reward = [], []
    mask, future_mask = [], []
    agent_id = []

    for traj in batch:
        state0.append(traj["state0"])
        state1.append(traj["state1"])
        action_indices.append(traj["action_indices"])
        reward.append(traj["reward"])
        mask.append(traj["mask"])
        future_state.append(traj["future_state"])
        future_action_indices.append(traj["future_action_indices"])
        future_reward.append(traj["future_reward"])
        future_mask.append(traj["future_mask"])
        agent_id.append(traj["attributes"]["agent_id"])
    
    return {
        "state0": torch.tensor(np.stack(state0), dtype=torch.float32),
        "state1": torch.tensor(np.stack(state1), dtype=torch.float32),
        "action_indices": torch.tensor(np.stack(action_indices), dtype=torch.long),
        "reward": torch.tensor(np.stack(reward), dtype=torch.float32),
        "mask": torch.tensor(np.stack(mask), dtype=torch.long),
        "future_state": torch.tensor(np.stack(future_state), dtype=torch.float32),
        "future_action_indices": torch.tensor(np.stack(future_action_indices), dtype=torch.long),
        "future_reward": torch.tensor(np.stack(future_reward), dtype=torch.float32),
        "future_mask": torch.tensor(np.stack(future_mask), dtype=torch.long),
        "agent_id": torch.tensor(np.stack(agent_id), dtype=torch.long),
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

class MultiAgentTrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for trajectories stored in an HDF5 file.
    Each trajectory is stored as a sample with states, actions, rewards, and additional attributes.

    Args:
        data_file_path (str): path to hdf5 dataset
        sequence_len (int): length of trajectory to return
        stride (int): when length of trajectory is longer than sequence_len, use sliding window of this stride to generate data
    """
    def __init__(
            self, 
            data_file_path: str,
            sequence_len: int = 30,
            stride: int = 1,    
        ):
        self.trajectories = [] 
        self.action_to_idx = {}  # Dictionary to store action name to index mapping
        self.idx_to_action = {}  # Dictionary to store index to action name mapping
        self.sequence_len = sequence_len
        self.stride = stride 

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
            # get max trajectory length
            max_traj_len = max([traj_group[traj_name].attrs["episode_len"] for traj_name in traj_group.keys()])
            max_future_traj_len = max_traj_len - self.sequence_len

            for traj_name in traj_group.keys():
                grp = traj_group[traj_name]
                traj_len = grp.attrs["episode_len"]
                # Read datasets stored in each trajectory group
                state0 = np.array(grp["state0"])
                state1 = np.array(grp["state1"])
                action_names = np.array(grp["action_name"], dtype=str)
                reward = np.array(grp["reward"])
                
                # Convert action names to indices using the predefined mapping
                action_indices = np.array([action_mapping.get(name, -1) for name in action_names], dtype=np.int64)
                
                # Read any attributes stored with the trajectory
                attributes = {attr: grp.attrs[attr] for attr in grp.attrs.keys()}

                pad_mode = "edge" # copy data for last step for padding (use "constant" for zero padding)

                # Store trajectories
                if traj_len < self.sequence_len:
                    # pad trajectories to sequence_len with last step 
                    pad_len = self.sequence_len - traj_len
                    pad_width = ((0, pad_len), (0, 0))
                    mask = np.zeros(self.sequence_len)
                    mask[:traj_len] = 1

                    # relabel actions for padded timesteps with "done" action (assume that agent stays at the goal after reaching it)
                    relabeled_action_indices = np.pad(action_indices, pad_width=(0,pad_len), mode="constant") # pad with zeros
                    relabeled_action_indices[sequence_len-pad_len:] = action_mapping["done"] # relabel with "done" action
                    # relabel padded timesteps as 
                    self.trajectories.append({
                        "state0": np.pad(state0, pad_width=pad_width, mode=pad_mode),
                        "state1": np.pad(state1, pad_width=pad_width),
                        # "action_indices": np.pad(action_indices, pad_width=(0,pad_len), mode=pad_mode),
                        "action_indices": relabeled_action_indices,
                        "action_names": np.pad(action_names, pad_width=(0,pad_len), mode=pad_mode),
                        "reward": np.pad(reward, pad_width=(0,pad_len), mode=pad_mode),
                        "mask": mask, # 1 if real data, 0 if padded data
                        "future_state": np.zeros((max_future_traj_len, state0.shape[1])), # there are no future state information
                        "future_action_indices": np.zeros(max_future_traj_len),
                        "future_reward": np.zeros(max_future_traj_len),
                        "future_mask": np.zeros(max_future_traj_len),
                        "attributes": attributes,
                        "traj_name": traj_name,
                    })
                else:
                    # use sliding window to generate samples
                    for i in range(0, traj_len-self.sequence_len+1, self.stride):
                        future_traj_len = traj_len - (i + self.sequence_len) 
                        pad_len = max_future_traj_len - future_traj_len # how many future steps to pad
                        pad_width = ((0, pad_len), (0, 0))
                        future_mask = np.zeros(max_future_traj_len)
                        future_mask[:future_traj_len] = 1
                        self.trajectories.append({
                            "state0": state0[i:i+self.sequence_len],
                            "state1": state1[i:i+self.sequence_len],
                            "action_indices": action_indices[i:i+self.sequence_len],
                            "action_names": action_names[i:i+self.sequence_len],
                            "reward": reward[i:i+self.sequence_len],
                            "mask": np.ones(self.sequence_len), # 1 if real data, 0 if padded data
                            "future_state": np.pad(state0[i+self.sequence_len:], pad_width=pad_width),
                            "future_action_indices": np.pad(action_indices[i+self.sequence_len:], pad_width=(0,pad_len)),
                            "future_reward": np.pad(reward[i+self.sequence_len:], pad_width=(0,pad_len)),
                            "future_mask": future_mask,
                            "attributes": attributes,
                            "traj_name": traj_name,
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
                              collate_fn=trajectory_dataset_collate_fn)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=trajectory_dataset_collate_fn)
    return train_loader, val_loader

def multiagent_traj_data_loaders(train_data, val_data, batch_size):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=mutliagent_traj_collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=mutliagent_traj_collate_fn,
    )
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
