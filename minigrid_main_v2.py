import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vqvae import RNNVQVAE, RNNFutureVQVAE, PredictionTypes
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
        for batch in train_loader:
            # Move data to device
            state0 = batch['state0'].to(device)
            action_indices = batch['action_indices'].to(device)
            reward = batch['reward'].to(device)
            mask = batch["mask"].to(device)
            future_state = batch["future_state"].to(device)
            future_action_indices = batch["future_action_indices"].to(device)
            future_mask = batch["future_mask"].to(device)
            
            for key in sampled_agents_train.keys():
                sampled_agents_train[key] += batch["agent_id"].tolist().count(key)

            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            
            # Forward pass
            logits, embedding_loss, pred_loss = model({
                "traj": x, # past trajectory
                "next_state": future_state[:,0,:], # current state
                "actions": action_indices,
                "future_actions": future_action_indices,  
                "mask": mask,
                "future_mask": future_mask,
            })

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

                for key in sampled_agents_valid.keys():
                    sampled_agents_valid[key] += batch["agent_id"].tolist().count(key)

                # Concatenate inputs
                x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
                
                # Forward pass
                logits, embedding_loss, pred_loss = model({
                    "traj": x, # past trajectory
                    "next_state": future_state[:,0,:], # current state
                    "actions": action_indices,
                    "future_actions" : future_action_indices,  
                    "mask": mask,
                    "future_mask": future_mask,
                })
                
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

        # print("sampled agents during training", sampled_agents_train)
        # print("sampled agents during validation", sampled_agents_valid)
        """
        LOGGING
        """
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
    
    # save final checkpoint
    torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt")
    torch.save(model.state_dict(), f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt")
    wandb.save(f"{wandb.run.dir}/optimizer_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
    wandb.save(f"{wandb.run.dir}/checkpoint_epoch_{epoch}.pt", base_path=wandb.run.dir, policy="now")
    
    return

def eval_vqvae(model, train_loader, val_loader, save_dir, device):

    model.eval()

    # get agent ids and minimum distance embedding indices for train and valid sets
    agent_ids = {}
    traj_encoding_indices = {}
    z_qs = {}

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    for split, loader in dataloaders.items():
        agent_ids[split] = []
        traj_encoding_indices[split] = []
        z_qs[split] = []

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

            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            
            # Get embeddings
            z_e, z_q, min_encodings, min_encoding_indices = model.get_embeddings({
                "traj": x,
                "mask": mask,
            })        

            # Get codebook usage per agent
            agent_ids[split].append(agent_id)
            min_encoding_indices = min_encoding_indices.cpu()
            # min_encoding_indices = min_encoding_indices.cpu().view(z_e.shape[0], z_e.shape[2])
            # traj_emb_ind = scipy.stats.mode(min_encoding_indices, axis=1)[0]
            # traj_encoding_indices[split].append(traj_emb_ind)
            z_qs[split].append(z_q.detach().cpu().numpy())
            traj_encoding_indices[split].append(min_encoding_indices)
    
        agent_ids[split] = torch.cat(agent_ids[split]).numpy()
        traj_encoding_indices[split] = np.vstack(traj_encoding_indices[split])
        z_qs[split] = np.vstack(z_qs[split])

        # plotting
        n_embs = model.vector_quantization.n_e # codebook size
        plot_codebook_usage_per_agent(
            token_ids=traj_encoding_indices[split], 
            agent_labels=agent_ids[split],
            n_embeddings=n_embs,
            savefile=os.path.join(save_dir, f"codebook_use_{split}.png")
        )
        plot_codebook_usage_heatmap(
            token_ids=traj_encoding_indices[split], 
            agent_labels=agent_ids[split], 
            n_embeddings=n_embs,
            savefile=os.path.join(save_dir, f"codebook_heatmap_{split}.png"),
        )
        plot_tsne(
            z_qs=z_qs[split],
            agent_labels=agent_ids[split],
            # savefile=os.path.join(save_dir, f"zq_tsne_{split}.png"),
            savename=os.path.join(save_dir, f"zq_tsne_{split}"),
        )

    # count the number of times each agent's trajectory maps to each embedding id
    unique_agent_ids = np.arange(6)
    unique_emb_ids = np.arange(n_embs)

    counts = {} # agent_id: np.array(# traj mapping to emb 0, # traj mapping to emb1, ..)

    for split in dataloaders.keys():
        counts[split] = {}
        for agent_id in unique_agent_ids:
            emb_id_mode = scipy.stats.mode(traj_encoding_indices[split], axis=1)[0]
            emb_counts = np.bincount(
                emb_id_mode[np.where(agent_ids[split] == agent_id)[0]],
                minlength=len(unique_emb_ids), 
            )
            counts[split][agent_id] = emb_counts

        # plot
        fig, ax = plt.subplots()
        bottom = np.zeros(n_embs)
        bar_width = 0.25
        for agent_id, cnts in counts[split].items():
            p = ax.bar(unique_emb_ids, cnts, label=f"agent {agent_id}", bottom=bottom)
            bottom += cnts
        ax.set_title(f"Agent ID to Emb ID Mapping ({split})")
        ax.set_xlabel("Agent ID")
        ax.set_ylabel("Codebook Usage Mode")
        ax.legend()
        plt.savefig(f"codebook_usage_mode_{split}.png")

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

def plot_codebook_usage_heatmap(token_ids, agent_labels, n_embeddings, savefile):
    unique_agents = sorted(set(agent_labels))
    agent_to_idx = {a: i for i, a in enumerate(unique_agents)}
    
    usage_matrix = np.zeros((len(unique_agents), n_embeddings))

    for i, agent in enumerate(agent_labels):
        codes, counts = np.unique(token_ids[i], return_counts=True)
        usage_matrix[agent_to_idx[agent], codes] += counts

    plt.figure(figsize=(12, len(unique_agents) * 0.6 + 3))
    sns.heatmap(usage_matrix, xticklabels=True, yticklabels=unique_agents, cmap="viridis")
    plt.xlabel("Codebook Index")
    plt.ylabel("Agent ID")
    plt.title("Heatmap of Codebook Usage per Agent")
    plt.savefig(savefile)
    plt.clf()

def plot_tsne(z_qs, agent_labels, savename):

    assert args.n_components in [2, 3], "visualization is only available for n_components = 2 or 3"

    z_qs_flattened = np.transpose(z_qs, (0, 2, 1)).reshape(z_qs.shape[0], -1) # flatten 
    
    # reduce dimensionality with PCA
    z_pca = PCA(n_components=100).fit_transform(z_qs_flattened) 
    
    # apply t-SNE
    z_tsne = TSNE(n_components=args.n_components, perplexity=5).fit_transform(z_pca)

    unique_agents = sorted(set(agent_labels))
    colors = plt.cm.tab10.colors
    agent_to_color = {agent: colors[i % len(colors)] for i, agent in enumerate(unique_agents)}

    fig = plt.figure(figsize=(8, 6))
    if args.n_components == 3:
        ax = fig.add_subplot(projection="3d")

    for agent in unique_agents:
        save_file = f"{savename}_{agent}.png"
        inds = np.where(agent_labels == agent)[0]
        tsne_latents = z_tsne[inds]
        if args.n_components == 2:
            plt.scatter(
                tsne_latents[:,0], tsne_latents[:,1], 
                color=agent_to_color[agent], 
                label=f"Agent {agent}",
            )
        elif args.n_components == 3:
            ax.scatter(
                tsne_latents[:,0], tsne_latents[:,1], tsne_latents[:,2],
                color=agent_to_color[agent],
                label=f"Agent {agent}",
            )
        plt.title(f"Trajectory-level z_q Embeddings (Flattened) {agent}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_file)

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
        dataset='MINIGRID', batch_size=args.batch_size, sequence_len=args.input_seq_len, balanced_sampling=args.balanced_sampling)

    # Initialize model
    model = RNNVQVAE(
        in_dim=args.in_dim,
        state_dim=args.state_dim,
        h_dim=args.h_dim,
        n_embeddings=args.n_embeddings,
        bidirectional=args.bidirectional,
        beta=args.beta,
        n_actions=args.n_actions,
        n_future_steps=args.n_future_steps,  
        n_past_steps=args.input_seq_len, 
        decoder_context_dim=args.decoder_context_dim,
        n_attention_heads=args.n_attention_heads,
        n_decoder_layers=args.n_decoder_layers,
    ).to(device)
    # model = RNNFutureVQVAE(
    #     in_dim=in_dim,
    #     state_dim=state_dim,
    #     h_dim=h_dim,
    #     n_embeddings=n_embeddings,
    #     bidirectional=bidirectional,
    #     beta=beta,
    #     n_actions=n_actions,
    #     n_steps=n_future_steps,
    #     decoder_context_dim=decoder_context_dim,
    #     n_attention_heads=n_attention_heads,
    #     n_decoder_layers=n_decoder_layers,
    # ).to(device)

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
    track: bool = True
    wandb_project_name: str = "human-knowledge-vqvae"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    plot_dir: str = "test"
    save_interval: int = 50 # number of epochs between each checkpoint saving

    """Dataset Settings"""
    input_seq_len: int = 30 # number of past steps to feed into encoder
    balanced_sampling: bool = True # if True, sample approximately equally from each agent
    
    """Encoder Settings"""
    in_dim: int = 88 # per-timestep feature dimension (len(obs + action + reward))
    h_dim: int = 64
    bidirectional: bool = True
    n_res_layers: int = 1
    n_embeddings: int = 6
    embedding_dim: int = 64
    
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