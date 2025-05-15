from torch.utils.data import Sampler
import random

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
