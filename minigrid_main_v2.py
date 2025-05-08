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

def train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    train_emb_losses = []
    train_pred_losses = []
    train_accuracy = []
    train_n_pred = []
    val_losses = []
    val_emb_losses = []
    val_pred_losses = []
    val_accuracy = []
    val_n_pred = []
    
    
    for epoch in range(num_epochs):
        model.train()

        # keep track of taining losses
        total_train_loss = 0
        total_emb_trian_loss = 0
        total_pred_train_loss = 0
        n_correct_train = 0
        n_pred_train = 0 # number of valid predictions (gt action exists)

        for batch in train_loader:
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
            loss = pred_loss + embedding_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_emb_trian_loss += embedding_loss.item()
            total_pred_train_loss += pred_loss.item()

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
            n_correct_train += (predicted_actions == gt_actions).sum().item()
            n_pred_train += gt_mask.sum().item()
            train_n_pred.append(n_pred_train)

        # Validation
        model.eval()

        # keep track of validation losses
        total_val_loss = 0
        total_emb_val_loss = 0
        total_pred_val_loss = 0
        n_correct_val = 0
        n_pred_val = 0 # number of valid predictions (gt action exists)

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
                
                loss = pred_loss + embedding_loss
                
                total_val_loss += loss.item()
                total_emb_val_loss += embedding_loss.item()
                total_pred_val_loss += pred_loss.item()
        
                # # record accuracy
                # future_mask = future_mask[:,:logits.shape[1]]
                # predicted_actions = logits.argmax(axis=-1)
                # gt_actions = future_action_indices[:,:logits.shape[1]]
                # gt_actions[future_mask == 0] = -100
                # n_correct_val += (predicted_actions == gt_actions).sum().item()
                # n_pred_val += future_mask.sum().item()

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
                n_correct_val += (predicted_actions == gt_actions).sum().item()
                n_pred_val += gt_mask.sum().item()
                val_n_pred.append(n_pred_val)

                # TODO - run autoregressive inference 

        avg_train_loss = total_train_loss / len(train_loader)
        avg_emb_train_loss = total_emb_trian_loss / len(train_loader)
        avg_pred_train_loss = total_pred_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_emb_val_loss = total_emb_val_loss / len(val_loader)
        avg_pred_val_loss = total_pred_val_loss / len(val_loader)
        avg_pred_train_accuracy = n_correct_train / n_pred_train
        avg_pred_val_accuracy = n_correct_val / n_pred_val
        
        train_losses.append(avg_train_loss)
        train_emb_losses.append(avg_emb_train_loss)
        train_pred_losses.append(avg_pred_train_loss)
        train_accuracy.append(avg_pred_train_accuracy)
        val_losses.append(avg_val_loss)
        val_emb_losses.append(avg_emb_val_loss)
        val_pred_losses.append(avg_pred_val_loss)
        val_accuracy.append(avg_pred_val_accuracy)

        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Train Loss (embedding): {avg_emb_train_loss:.4f}, '
            f'Train Loss (prediction): {avg_pred_train_loss:.4f}, '
            f'Train Accuracy: {avg_pred_train_accuracy:.4f}, '
            
            f'Val Loss: {avg_val_loss:.4f}, '
            f'Val Loss (embedding): {avg_emb_val_loss:.4f}, '
            f'Val Loss (prediction): {avg_pred_val_loss:.4f}, '
            f'Val Accuracy: {avg_pred_val_accuracy:.4f}, '
            # f'Perplexity: {perplexity:.4f}'
        )
    
    return (train_losses, train_emb_losses, train_pred_losses, train_accuracy, train_n_pred,
        val_losses, val_emb_losses, val_pred_losses, val_accuracy, val_n_pred)

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
        save_file = f"{savename}_{agent}.png"
        inds = np.where(agent_labels == agent)[0]
        tsne_latents = z_tsne[inds]
        plt.scatter(
            tsne_latents[:,0], tsne_latents[:,1], 
            color=agent_to_color[agent], 
            label=f"Agent {agent}",
        )
        plt.title(f"Trajectory-level z_q Embeddings (Flattened) {agent}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_file)

def main(args):
    ####### Hyperparameters #######    
    """Dataset Settings"""
    input_seq_len = 30 # number of past steps to feed into encoder
    """Encoder Settings"""
    in_dim = 88 # per-timestep feature dimension (len(obs + action + reward))
    h_dim = 64
    bidirectional = True
    n_res_layers = 1
    n_embeddings = 6
    embedding_dim = 64
    """Decoder Settings"""
    state_dim = 86 # dimension of state the decoder sees as input
    n_actions = 3 # number of discrete actions
    n_future_steps = 10
    decoder_context_dim = 128
    n_attention_heads = 4
    n_decoder_layers = 2
    """Training Settings"""
    beta = 0.25
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-4
    ####### Hyperparameters #######    

    # make directory to save result plots
    save_dir = os.path.join("results", args.plot_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    training_data, validation_data, train_loader, val_loader, x_train_var = load_data_and_data_loaders(
        dataset='MINIGRID', batch_size=batch_size, sequence_len=input_seq_len)

    # Initialize model
    model = RNNVQVAE(
        in_dim=in_dim,
        state_dim=state_dim,
        h_dim=h_dim,
        n_embeddings=n_embeddings,
        bidirectional=bidirectional,
        beta=beta,
        n_actions=n_actions,
        n_future_steps=n_future_steps,  
        n_past_steps=input_seq_len, 
        decoder_context_dim=decoder_context_dim,
        n_attention_heads=n_attention_heads,
        n_decoder_layers=n_decoder_layers,
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
    train_losses, train_emb_losses, train_pred_losses, train_accuracy, train_n_pred, \
        val_losses, val_emb_losses, val_pred_losses, val_accuracy, val_n_pred \
            = train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device)
    eval_vqvae(model, train_loader, val_loader, save_dir, device)

    # Save results
    timestamp = readable_timestamp()
    results = {
        'train_losses': train_losses,
        'train_emb_losses': train_emb_losses, 
        'train_pred_losses': train_pred_losses, 
        'val_losses': val_losses,
        'val_emb_losses': val_emb_losses, 
        'val_pred_losses':val_pred_losses,
        'x_train_var': x_train_var
    }
    hyperparameters = {
        'h_dim': h_dim,
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
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.plot(train_emb_losses, label='Training Loss')
    plt.plot(val_emb_losses, label='Validation Loss')
    plt.title('Training and Validation Losses (Embedding)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "emb_loss_plot.png"))
    plt.clf()
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_pred_losses, label='Training Loss')
    plt.plot(val_pred_losses, label='Validation Loss')
    plt.title('Training and Validation Losses (CE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pred_loss_plot.png'))
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label='Training')
    plt.plot(val_accuracy, label='Validation')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.plot(train_n_pred, label="Training")
    plt.plot(val_n_pred, label="Validation")
    plt.title('Number of valid action predictions')
    plt.xlabel('Epoch')
    plt.ylabel('N valid predictions')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'n_valid_predictions.png'))
    plt.clf()

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-dir", type=str, help="directory to save result plots to")
    args = parser.parse_args()

    main(args) 