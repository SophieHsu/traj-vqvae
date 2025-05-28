import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vqvae import GumbelCodebookFreeVAE, PredictionTypes
from utils import load_data_and_data_loaders, save_model_and_results, readable_timestamp

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import argparse
import wandb
import tyro
import time
import random
from tqdm import tqdm

from dataclasses import dataclass

def train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device):

    # setup wandb logging
    run_name = f"{args.exp_name}_{args.seed}_{args.plot_dir}_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        if wandb.run is not None:
            wandb.run.log_code(".")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_n_pred, val_n_pred = [], []
    
    for epoch in tqdm(range(num_epochs)):

        """
        TRAINING
        """
        model.train()

        # keep track of taining losses
        total_train_loss = 0
        total_emb_train_loss_raw = 0
        total_pred_train_loss_raw = 0
        total_emb_train_loss_weighted = 0
        total_pred_train_loss_weighted = 0
        n_correct_train = 0
        n_pred_train = 0 # number of valid predictions (gt action exists)
        n_correct_past_train = 0
        n_pred_past_train = 0
        n_correct_future_train = 0
        n_pred_future_train = 0

        sampled_agents_train = {i: 0 for i in range(6)}
        sampled_agents_valid = {i: 0 for i in range(6)}

        codebook_usage = []
        agent_ids = []

        for batch in train_loader:
            # Move data to device
            state0 = batch['state0'].to(device)
            action_indices = batch['action_indices'].to(device)
            reward = batch['reward'].to(device)
            mask = batch["mask"].to(device)
            future_state = batch["future_state"].to(device)
            future_action_indices = batch["future_action_indices"].to(device)
            future_mask = batch["future_mask"].to(device)
            agent_ids.append(batch["agent_id"])
            
            for key in sampled_agents_train.keys():
                sampled_agents_train[key] += batch["agent_id"].tolist().count(key)

            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            
            # Forward pass
            logits, embedding_loss, pred_loss, min_encodings, min_encoding_indices = model({
                "traj": x, # past trajectory
                "next_state": future_state[:,0,:], # current state
                "actions": action_indices,
                "future_actions": future_action_indices,  
                "mask": mask,
                "future_mask": future_mask,
            })

            masked_codebook_use = min_encoding_indices.cpu().view(x.shape[0], x.shape[1]) # (B, T) recover temporal dimension
            masked_codebook_use[mask == 0] = -100 # replace padded timesteps with placeholder id
            codebook_usage.append(masked_codebook_use) # (B, T)

            # Total loss
            pred_loss_weighted = args.prediction_loss_weight * pred_loss
            embedding_loss_weighted = args.embedding_loss_weight * embedding_loss
            loss = pred_loss_weighted + embedding_loss_weighted
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_emb_train_loss_raw += embedding_loss.item()
            total_pred_train_loss_raw += pred_loss.item()
            total_emb_train_loss_weighted += embedding_loss_weighted.item()
            total_pred_train_loss_weighted += pred_loss_weighted.item()
            
            # record accuracy
            predicted_actions = logits.argmax(axis=-1)
            if model.prediction_type == PredictionTypes.FUTURE:
                gt_mask = future_mask[:,:logits.shape[1]]
                gt_actions = future_action_indices[:,:logits.shape[1]]
                gt_actions[gt_mask == 0] = -100
                n_pred_future_train += gt_mask.sum().item()
                n_correct_future_train += (predicted_actions == gt_actions).sum().item()
            elif model.prediction_type == PredictionTypes.PAST_FUTURE:
                assert action_indices.shape[1] == model.n_past_steps, "make sure n_past_steps and the number of actions in the trajectory matches"
                gt_actions = torch.cat((action_indices, future_action_indices[:,:model.n_future_steps]), dim=1)
                gt_mask = torch.cat((mask, future_mask[:,:model.n_future_steps]), dim=1)
                gt_actions[gt_mask == 0] = -100
                # breakpoint()
                n_pred_past_train += gt_mask[:model.n_past_steps].sum().item()
                n_correct_past_train += (predicted_actions[:model.n_past_steps] == gt_actions[:model.n_past_steps]).sum().item()
                n_pred_future_train += gt_mask[model.n_past_steps:].sum().item()
                n_correct_future_train += (predicted_actions[model.n_past_steps:] == gt_actions[model.n_past_steps:]).sum().item()
            n_correct_train += (predicted_actions == gt_actions).sum().item()
            n_pred_train += gt_mask.sum().item()
            train_n_pred.append(n_pred_train)

        # compute codebook usage entropy per agent
        codebook_use_entropy = compute_codebook_use_entropy(
            codebook_ids=torch.cat(codebook_usage, dim=0),
            agent_labels=torch.cat(agent_ids),
            codebook_size=args.n_embeddings,
        )

        if args.track:
            wandb.run.log({f"{key}_entropy_train": codebook_use_entropy[key] for key in codebook_use_entropy.keys()})

        """
        EVALUATION
        """
        model.eval()
        
        # keep track of validation losses
        total_val_loss = 0
        total_emb_val_loss_raw = 0
        total_pred_val_loss_raw = 0
        total_emb_val_loss_weighted = 0
        total_pred_val_loss_weighted = 0
        n_correct_val = 0
        n_pred_val = 0 # number of valid predictions (gt action exists)\
        n_correct_past_val = 0
        n_pred_past_val = 0
        n_correct_future_val = 0
        n_pred_future_val = 0

        codebook_usage = []
        agent_ids = []

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                state0 = batch['state0'].to(device)
                state1 = batch['state1'].to(device)
                action_indices = batch['action_indices'].to(device)
                reward = batch['reward'].to(device)
                mask = batch["mask"].to(device)
                future_state = batch["future_state"].to(device)
                future_action_indices = batch["future_action_indices"].to(device)
                # future_rewards  = batch["future_reward"].to(device)
                future_mask = batch["future_mask"].to(device)
                agent_ids.append(batch["agent_id"])

                for key in sampled_agents_valid.keys():
                    sampled_agents_valid[key] += batch["agent_id"].tolist().count(key)

                # Concatenate inputs
                x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
                
                # Forward pass
                logits, embedding_loss, pred_loss, min_encodings, min_encoding_indices = model({
                    "traj": x, # past trajectory
                    "next_state": future_state[:,0,:], # current state
                    "actions": action_indices,
                    "future_actions" : future_action_indices,
                    "mask": mask,
                    "future_mask": future_mask,
                })

                masked_codebook_use = min_encoding_indices.cpu().view(x.shape[0], x.shape[1]) # (B, T) recover temporal dimension
                masked_codebook_use[mask == 0] = -100 # replace padded timesteps with placeholder id
                codebook_usage.append(masked_codebook_use) # (B, T)
                
                pred_loss_weighted = args.prediction_loss_weight * pred_loss
                embedding_loss_weighted = args.embedding_loss_weight * embedding_loss
                loss = pred_loss_weighted + embedding_loss_weighted
                
                total_val_loss += loss.item()
                total_emb_val_loss_raw += embedding_loss.item()
                total_pred_val_loss_raw += pred_loss.item()
                total_emb_val_loss_weighted += embedding_loss_weighted.item()
                total_pred_val_loss_weighted += pred_loss_weighted.item()

                # record accuracy
                predicted_actions = logits.argmax(axis=-1)
                if model.prediction_type == PredictionTypes.FUTURE:
                    gt_mask = future_mask[:,:logits.shape[1]]
                    gt_actions = future_action_indices[:,:logits.shape[1]]
                    gt_actions[gt_mask == 0] = -100
                elif model.prediction_type == PredictionTypes.PAST_FUTURE:
                    assert action_indices.shape[1] == model.n_past_steps, "make sure n_past_steps and the number of actions in the trajectory matches"
                    gt_actions = torch.cat((action_indices, future_action_indices[:,:model.n_future_steps]), dim=1)
                    gt_mask = torch.cat((mask, future_mask[:,:model.n_future_steps]), dim=1)
                    gt_actions[gt_mask == 0] = -100
                    n_pred_past_val += gt_mask[:model.n_past_steps].sum().item()
                    n_correct_past_val += (predicted_actions[:model.n_past_steps] == gt_actions[:model.n_past_steps]).sum().item()
                    n_pred_future_val += gt_mask[model.n_past_steps:].sum().item()
                    n_correct_future_val += (predicted_actions[model.n_past_steps:] == gt_actions[model.n_past_steps:]).sum().item()
                n_correct_val += (predicted_actions == gt_actions).sum().item()
                n_pred_val += gt_mask.sum().item()
                val_n_pred.append(n_pred_val)

        # compute codebook usage entropy per agent
        codebook_use_entropy = compute_codebook_use_entropy(
            codebook_ids=torch.cat(codebook_usage, dim=0),
            agent_labels=torch.cat(agent_ids),
            codebook_size=args.n_embeddings,
        )
        if args.track:
            wandb.run.log({f"{key}entropy_valid": codebook_use_entropy[key] for key in codebook_use_entropy.keys()})

        # print("sampled agents during training", sampled_agents_train)
        # print("sampled agents during validation", sampled_agents_valid)
        """
        LOGGING
        """
        if args.track:
            wandb.run.log({
                "train_loss": total_train_loss / len(train_loader),

                "train_loss_embedding_raw": total_emb_train_loss_raw / len(train_loader),
                "train_loss_prediction_raw": total_pred_train_loss_raw / len(train_loader),

                "train_loss_embedding_weighted": total_emb_train_loss_weighted / len(train_loader),
                "train_loss_prediction_weighted": total_pred_train_loss_weighted / len(train_loader),

                "valid_loss": total_val_loss / len(val_loader),

                "valid_loss_embedding_raw": total_emb_val_loss_raw / len(val_loader),
                "valid_loss_prediction_raw": total_pred_val_loss_raw / len(val_loader),
                
                "valid_loss_embedding_weighted": total_emb_val_loss_weighted / len(val_loader),
                "valid_loss_prediction_weighted": total_pred_val_loss_weighted / len(val_loader),
                
                "train_accuracy": n_correct_train / n_pred_train,
                "train_accuracy_past": n_correct_past_train / n_pred_past_train,
                "train_accuracy_future": n_correct_future_train / n_pred_future_train,
                "valid_accuracy": n_correct_val / n_pred_val,
                "valid_accuracy_past": n_correct_past_val / n_pred_past_val,
                "valid_accyracy_future": n_correct_future_val / n_pred_future_val,
            })
    
            # save model checkpoint
            if epoch > 0 and (epoch % args.save_interval) == 0:
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt")
                torch.save(model.state_dict(), f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt")
                wandb.save(f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
    
    if args.track:
        # save final checkpoint
        torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt")
        torch.save(model.state_dict(), f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt")
        wandb.save(f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
        wandb.save(f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
    
    return

def eval_vqvae(model, train_loader, val_loader, save_dir, device):

    model.eval()

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    for split, loader in dataloaders.items():
        agent_ids = []
        codebook_usage = []
        z_qs = []
        masks = []

        for batch in loader:
            # Move data to device
            state0 = batch['state0'].to(device)
            state1 = batch['state1'].to(device)
            action_indices = batch['action_indices'].to(device)
            reward = batch['reward'].to(device)
            mask = batch["mask"].to(device)
            future_state = batch["future_state"].to(device)
            future_action_indices = batch["future_action_indices"].to(device)
            future_mask = batch["future_mask"].to(device)
            agent_id = batch["agent_id"]
            agent_ids.append(agent_id)

            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            
            # Get embeddings
            z_e, z_q, min_encodings, min_encoding_indices = model.get_embeddings({
                "traj": x,
                "mask": mask,
            })        

            # record codebook usage and z_q's
            masked_codebook_use = min_encoding_indices.cpu().view(x.shape[0], x.shape[1]) # recover temporal dimension
            masked_codebook_use[mask == 0] = -100 # replace padded timesteps with placeholder id
            codebook_usage.append(masked_codebook_use) #  (B, T)
            z_qs.append(z_q.detach().cpu())
            masks.append(mask.cpu())

        agent_ids = torch.cat(agent_ids)
        codebook_usage = torch.cat(codebook_usage, dim=0)
        z_qs = torch.cat(z_qs, dim=0)
        masks = torch.cat(masks, dim=0)

        unique_agents = set(agent_ids.tolist())
        per_agent_data = {}
        for unique_agent in unique_agents:
            per_agent_data[unique_agent] = {
                "codebook_ids": codebook_usage[agent_ids == unique_agent],
                "z_q": z_qs[agent_ids == unique_agent],
                "mask": masks[agent_ids == unique_agent],
            }
        
        plot_codebook_usage_heatmap(
            per_agent_data=per_agent_data,
            n_embeddings=args.n_embeddings,
            savefile=os.path.join(save_dir, f"codebok_usage_{split}.png"),
        )

        plot_tsne(per_agent_data=per_agent_data, savename=os.path.join(save_dir, f"zq_tsne_{split}"))

def plot_codebook_usage_heatmap(per_agent_data, n_embeddings, savefile, ignore_id=-100):
    """
    per_agent_data (dict): {agent_id: {"codebook_ids" : (N, T) with codebook indices}}
        where agent_id = 0, 1, 2, ...
    """
    unique_agents = list(per_agent_data.keys())
    usage_matrix = np.zeros((len(unique_agents), n_embeddings))

    for agent in unique_agents:
        codebook_ids = per_agent_data[agent]["codebook_ids"]
        codes, counts = np.unique(codebook_ids, return_counts=True)
        # handle code to ignore
        ignore_idx = np.where(codes == ignore_id)[0]
        if ignore_idx.size > 0:
            codes = np.delete(codes, ignore_idx)
            counts = np.delete(counts, ignore_idx)
        usage_matrix[agent, codes] += counts

    plt.figure(figsize=(12, len(unique_agents) * 0.6 + 3))
    sns.heatmap(usage_matrix, xticklabels=True, yticklabels=unique_agents, cmap="viridis")
    plt.xlabel("Codebook Index")
    plt.ylabel("Agent ID")
    plt.title("Heatmap of Codebook Usage per Agent")
    plt.savefig(savefile)
    plt.clf()

def plot_tsne(per_agent_data, savename):
    """
    per_agent_data (dict): 
        {agent_id: {"z_q": (N, T, d_embedding),
                        "mask": (N, T)}}
    """
    unique_agents = list(per_agent_data.keys())
    mean_zqs = []
    last_zqs = []
    agent_ids = []

    for agent in unique_agents:
        mask = per_agent_data[agent]["mask"]
        z_q = per_agent_data[agent]["z_q"]
        agent_ids += [agent]*z_q.shape[0]
        
        # masked mean across temporal dimension
        z_q_masked = z_q * mask.unsqueeze(-1)
        sum_z_q = z_q_masked.sum(dim=1)  # (B, D)
        valid_counts = mask.sum(dim=1, keepdim=True)  # (B, 1)
        mean_z_q = sum_z_q / (valid_counts + 1e-6)  # (B, D)
        mean_zqs.append(mean_z_q)

        # last valid timestep
        last_real_index = mask.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), axis=1)
        z_q_last = z_q[torch.arange(z_q.shape[0]), last_real_index]
        last_zqs.append(z_q_last)

        # TODO - is there a way to keep all?

    agent_ids = np.array(agent_ids)
    mean_zqs = torch.cat(mean_zqs, dim=0)
    last_zqs = torch.cat(last_zqs, dim=0)
    zq_pca_masked_mean = PCA(n_components=20).fit_transform(mean_zqs.numpy())
    zq_2d_masked_mean = TSNE(n_components=2, perplexity=30).fit_transform(zq_pca_masked_mean)
    zq_pca_last = PCA(n_components=20).fit_transform(last_zqs.numpy())
    zq_2d_last = TSNE(n_components=2, perplexity=30).fit_transform(zq_pca_last)
    
    # plotting
    colors = plt.cm.tab10.colors
    agent_to_color = {agent: colors[i % len(colors)] for i, agent in enumerate(unique_agents)}
    
    # masked mean plots
    plt.figure(figsize=(8, 6))
    for agent_id in unique_agents:
        idx = (agent_ids == agent_id)
        plt.scatter(
            zq_2d_masked_mean[idx, 0], zq_2d_masked_mean[idx, 1], 
            color=agent_to_color[agent_id],
            label=f'Agent {agent_id}', 
            alpha=0.7,
        )
        plt.title(f"Masked Mean z_qs {agent_id}")
        plt.legend()
        plt.tight_layout()
        save_file = f"{savename}_masked_mean_{agent_id}.png"
        plt.savefig(save_file)

    plt.clf()

    # last step plots
    plt.figure(figsize=(8, 6))
    for agent_id in unique_agents:
        idx = (agent_ids == agent_id)
        plt.scatter(
            zq_2d_last[idx, 0], zq_2d_last[idx, 1], 
            color=agent_to_color[agent_id],
            label=f'Agent {agent_id}', 
            alpha=0.7,
        )
        plt.title(f"Last step z_qs {agent_id}")
        plt.legend()
        plt.tight_layout()
        save_file = f"{savename}_last_step_{agent_id}.png"
        plt.savefig(save_file)

    plt.clf()


def compute_codebook_use_entropy(codebook_ids, agent_labels, codebook_size, ignore_code=-100):
    entropies = {}
    unique_agents = set(agent_labels.tolist())
    max_entropy = torch.log2(torch.tensor(codebook_size, dtype=torch.float))
    for id in unique_agents:
        codebook_usage = codebook_ids[agent_labels == id].flatten() # get all codebook indices used for this agent
        codebook_usage = codebook_usage[codebook_usage != ignore_code] # remove the codes to ignore (padded steps)
        counts = torch.bincount(torch.tensor(codebook_usage), minlength=codebook_size).float()
        prob = counts / counts.sum()
        non_zero_prob = prob[prob > 0] # filter out 0 probability codes to avoid zero division
        entropy = -(non_zero_prob * torch.log2(non_zero_prob)).sum().item()
        entropies[id] = entropy / max_entropy
    entropies["average"] = np.array([val for val in entropies.values()]).mean()
    return entropies

def plot_codebook_usage_per_agent(token_ids, agent_labels, n_embeddings, savefile):
    agent_to_tokens = defaultdict(list)

    for i, agent_id in enumerate(agent_labels):
        agent_to_tokens[agent_id].extend(token_ids[i].tolist())

    plt.figure(figsize=(16, 5 * len(agent_to_tokens)))
    for idx, (agent_id, tokens) in enumerate(agent_to_tokens.items()):
        plt.subplot(len(agent_to_tokens), 1, idx + 1)
        sns.histplot(tokens, bins=n_embeddings, discrete=True, stat='density')
        plt.title(f"Codebook Usage Histogram - Agent {agent_id}")
        plt.xlabel("Codebook Index")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(savefile)
    plt.clf()

def main(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # make directory to save result plots
    save_dir = os.path.join("results", args.plot_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    training_data, validation_data, train_loader, val_loader, x_train_var = load_data_and_data_loaders(
        dataset='MINIGRID', batch_size=args.batch_size, 
        sequence_len=args.input_seq_len, balanced_sampling=args.balanced_sampling,
        data_file=args.data_file,
    )

    # Initialize model
    model = GumbelCodebookFreeVAE(
        in_dim=args.in_dim,
        state_dim=args.state_dim,
        h_dim=args.h_dim,
        n_embeddings=args.n_embeddings,
        bidirectional=args.bidirectional,
        gumbel_temperature=args.gumbel_temperature,
        n_actions=args.n_actions,
        n_past_steps=args.input_seq_len,
        n_future_steps=args.n_future_steps,
        decoder_context_dim=args.decoder_context_dim,
        n_attention_heads=args.n_attention_heads,
        n_decoder_layers=args.n_decoder_layers,
    ).to(device)
    
    # Train model
    print("Starting training...")
    train_vqvae(model, train_loader, val_loader, args.num_epochs, args.learning_rate, device)
    eval_vqvae(model, train_loader, val_loader, save_dir, device)
  
@dataclass
class Args:
    """Torch, cuda, seed"""
    exp_name: str = "vqvae"
    seed: int = 1 # NOTE 
    torch_deterministic: bool = True
    cuda: bool = True
    
    """Logging"""
    track: bool = False
    wandb_project_name: str = "human-knowledge-vqvae"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    plot_dir: str = "test"
    save_interval: int = 50 # number of epochs between each checkpoint saving

    """Dataset Settings"""
    data_file: str = "data/minigrid/9-rooms-2.5k/6agents.hdf5"
    input_seq_len: int = 30 # number of past steps to feed into encoder
    balanced_sampling: bool = True # if True, sample approximately equally from each agent
    
    """Encoder Settings"""
    in_dim: int = 88 # per-timestep feature dimension (len(obs + action + reward))
    h_dim: int = 64
    bidirectional: bool = True
    n_res_layers: int = 1
    n_embeddings: int = 6
    embedding_dim: int = 64
    gumbel_temperature: float = 1.0 # 1.0 is one-hot, 0.0 is uniform
    
    """Decoder Settings"""
    state_dim: int = 86 # dimension of state the decoder sees as input
    n_actions: int = 3 # number of discrete actions
    n_future_steps: int = 10
    decoder_context_dim: int = 128
    n_attention_heads: int = 4
    n_decoder_layers: int = 2
    
    """Training Hyperparams"""
    beta: float = 0.25
    num_epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-4 
    embedding_loss_weight: float = 1.0 
    prediction_loss_weight: float = 1.0

    """Visualization Settings"""
    n_components: int = 2 # number of components to flatten zq for visualization (2 or 3)  

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 