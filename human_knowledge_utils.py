from torch.utils.data import Sampler
import torch
import random

from enum import Enum

class Actions(Enum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6

# actions allowed for each agent
AGENT_ACTION_MASKS = torch.tensor([
    [1, 1, 1], # agent 0
    [1, 1, 0], # agent 1
    [1, 0, 1], # agent 2
    [1, 1, 1], # agent 3
    [1, 1, 0], # agent 4
    [1, 0, 1], # agent 5
])
# whether each agent is partial or full observation
AGENT_OBS_TYPES = torch.tensor([1, 1, 1, 0, 0, 0])

def misclassification_validity_check(
    preds,                  # (B,)
    true_labels,            # (B,)
    action_trajs,           # (B, T)
    n_invisible_collisions, # (B,)
):
    B, T = action_trajs.shape
    A = AGENT_ACTION_MASKS.shape[1]

    # handle padded steps (where action is -100)
    valid_mask = (action_trajs >= 0)  # shape: (B, T), bool
    action_trajs_clipped = action_trajs.clone()
    action_trajs_clipped[~valid_mask] = 0  # replace with 0 temporarily

    action_trajs_onehot = torch.nn.functional.one_hot(action_trajs_clipped, num_classes=A)  # (B, T, A)

    # mask out padded steps
    valid_mask_exp = valid_mask.unsqueeze(-1)  # (B, T, 1)
    action_trajs_onehot = action_trajs_onehot * valid_mask_exp  # (B, T, A)

    action_taken_mask = action_trajs_onehot.any(dim=1).int()  # (B, A)

    # Look up predicted and true agent action masks (B, A)
    pred_action_mask = AGENT_ACTION_MASKS[preds]      # (B, A)
    true_action_mask = AGENT_ACTION_MASKS[true_labels]  # (B, A)

    # Invalid condition 1: took actions not allowed by predicted agent
    invalid_1 = ((action_taken_mask - pred_action_mask) > 0).any(dim=1)  # (B,)

    # Look up predicted agent observability
    pred_obs_type = AGENT_OBS_TYPES[preds]  # (B,)  (0=partial, 1=full)

    # Invalid condition 2: predicted agent is full obs, but invisible collisions occurred
    invalid_2 = (pred_obs_type == 1) & (n_invisible_collisions > 0)  # (B,)

    # Combine invalid conditions
    invalid_mask = invalid_1 | invalid_2  # (B,)

    # Correct prediction mask
    correct_mask = preds == true_labels  # (B,)

    # Valid misclassifications (only if not correct or invalid)
    # Find actions allowed by true agent but not by predicted agent (B, A)
    additional_actions = (true_action_mask & (~pred_action_mask.bool())).int()  # (B, A)

    # If agent *did* take any of those additional actions → invalid
    took_additional_action = (action_taken_mask & additional_actions).any(dim=1)  # (B,)

    valid_mask = (~correct_mask) & (~invalid_mask) & (~took_additional_action)

    # Compose final validity label
    validity = torch.full((B,), fill_value=-1, dtype=torch.int8)
    validity[correct_mask] = 1
    validity[valid_mask] = 0  # valid misclassification

    return {
        "validity": validity.tolist(),
        "n_total": B,
        "n_correct": correct_mask.sum().item(),
        "n_invalid_incorrect": invalid_mask.sum().item(),
        "n_valid_incorrect": valid_mask.sum().item(),
        "n_incorrect": (valid_mask | invalid_mask).sum().item(),
    }

def get_agent_to_indices(dataset):
    """
    Get dictionary mapping agent id to the datapoint indices corresponding to each agent
    """
    agent_to_data_idx = {}
    for i, data in enumerate(dataset):
        agent_id = data["agent_id"]
        if agent_id not in agent_to_data_idx.keys():
            agent_to_data_idx[agent_id] = []
        agent_to_data_idx[agent_id].append(i)
    return agent_to_data_idx

class BalancedAgentSampler(Sampler):
    def __init__(self, agent_to_indices, num_samples=None):
        self.agent_to_indices = agent_to_indices
        self.agent_ids = list(agent_to_indices.keys())
        self.num_samples = num_samples or min(len(idxs) for idxs in agent_to_indices.values()) * len(agent_to_indices)

    def __iter__(self):
        samples = []
        per_agent = self.num_samples // len(self.agent_ids)

        for agent_id in self.agent_ids:
            agent_indices = self.agent_to_indices[agent_id]
            sampled = random.choices(agent_indices, k=per_agent)  # use choices for replacement, sample for without
            samples.extend(sampled)

        random.shuffle(samples)
        return iter(samples)

    def __len__(self):
        return self.num_samples
