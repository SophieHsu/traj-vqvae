import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vqvae import RNNVQVAE
from utils import load_data_and_data_loaders, save_model_and_results, readable_timestamp
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            # Move data to device
            state0 = batch['state0'].to(device)
            state1 = batch['state1'].to(device)
            action_indices = batch['action_indices'].to(device)
            reward = batch['reward'].to(device)
            mask = batch["mask"].to(device)
            
            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)

            # Forward pass
            # embedding_loss, x_hat, perplexity = model({"traj": x, "mask": mask})
            model({"traj": x, "mask": mask})
            
            # Compute reconstruction loss with state1 as target
            recon_loss = nn.MSELoss()(x_hat, x.reshape(x.shape[0], 1, -1))
            
            # Total loss
            loss = recon_loss + embedding_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                state0 = batch['state0'].to(device)
                state1 = batch['state1'].to(device)
                action_indices = batch['action_indices'].to(device)
                reward = batch['reward'].to(device)
                
                # Concatenate inputs
                x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
                
                embedding_loss, x_hat, perplexity = model(x)
                recon_loss = nn.MSELoss()(x_hat, x.reshape(x.shape[0], 1, -1))
                loss = recon_loss + embedding_loss
                
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Perplexity: {perplexity:.4f}')
    
    return train_losses, val_losses

def eval_vqvae(model, train_loader, val_loader, device):

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
            agent_id = batch["agent_id"]
            
            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)

            # Encode
            z_e = model.encoder(x.reshape(x.shape[0], 1, -1))
            z_e = model.pre_quantization_conv(z_e)
            _, z_q, _, min_encodings, min_encoding_indices = model.vector_quantization(z_e)
            agent_ids[split].append(agent_id)
            min_encoding_indices = min_encoding_indices.cpu().view(z_e.shape[0], z_e.shape[2])
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
            savefile=f"codebook_use_{split}.png"
        )
        plot_codebook_usage_heatmap(
            token_ids=traj_encoding_indices[split], 
            agent_labels=agent_ids[split], 
            n_embeddings=n_embs,
            savefile=f"codebook_heatmap_{split}.png",
        )
        plot_tsne(
            z_qs=z_qs[split],
            agent_labels=agent_ids[split],
            savefile=f"zq_tsne_{split}.png",
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

def plot_tsne(z_qs, agent_labels, savefile):
    z_qs_flattened = np.transpose(z_qs, (0, 2, 1)).reshape(z_qs.shape[0], -1) # flatten 
    
    # reduce dimensionality with PCA
    z_pca = PCA(n_components=100).fit_transform(z_qs_flattened) 
    
    # apply t-SNE
    z_tsne = TSNE(n_components=2, perplexity=5).fit_transform(z_pca)

    unique_agents = sorted(set(agent_labels))
    colors = plt.cm.tab10.colors
    agent_to_color = {agent: colors[i % len(colors)] for i, agent in enumerate(unique_agents)}

    plt.figure(figsize=(8, 6))
    for agent in unique_agents:
        inds = np.where(agent_labels == agent)[0]
        tsne_latents = z_tsne[inds]
        plt.scatter(
            tsne_latents[:,0], tsne_latents[:,1], 
            color=agent_to_color[agent], 
            label=f"Agent {agent}",
        )
    # for i, (x, y) in enumerate(z_tsne):
    #     breakpoint()
    #     agent = agent_labels[i]
    #     plt.scatter(x, y, color=agent_to_color[agent], label=f"Agent {agent}" if agent not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.title("Trajectory-level z_q Embeddings (Flattened)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile)

def main():
    # Hyperparameters
    in_dim = 88 # per-timestep feature dimension (len(obs + action + reward))
    # h_dim = 128
    h_dim = 64
    res_h_dim = 32
    n_rnn_layers = 1
    bidirectional = True
    # res_h_dim = 16
    n_res_layers = 1
    n_embeddings = 6
    embedding_dim = 64
    # embedding_dim = 16
    beta = 0.25
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    training_data, validation_data, train_loader, val_loader, x_train_var = load_data_and_data_loaders(
        dataset='MINIGRID', batch_size=batch_size)

    # Initialize model
    # model = RNNVQVAE(in_dim, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta).to(device)
    model = RNNVQVAE(in_dim, h_dim, n_embeddings, bidirectional, beta).to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device)

    eval_vqvae(model, train_loader, val_loader, device)

    # Save results
    timestamp = readable_timestamp()
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'x_train_var': x_train_var
    }
    hyperparameters = {
        'h_dim': h_dim,
        'res_h_dim': res_h_dim,
        'n_res_layers': n_res_layers,
        'n_embeddings': n_embeddings,
        'embedding_dim': embedding_dim,
        'beta': beta,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    save_model_and_results(model, results, hyperparameters, timestamp)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/loss_plot_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    main() 