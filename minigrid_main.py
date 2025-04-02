import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from utils import load_data_and_data_loaders, save_model_and_results, readable_timestamp
import numpy as np
import matplotlib.pyplot as plt
import os

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
            
            # Concatenate inputs
            x = torch.cat([state0, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            
            # Forward pass
            embedding_loss, x_hat, perplexity = model(x)
            
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

def main():
    # Hyperparameters
    h_dim = 128
    res_h_dim = 32
    n_res_layers = 1
    n_embeddings = 2
    embedding_dim = 64
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
    model = VQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta).to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_vqvae(model, train_loader, val_loader, num_epochs, learning_rate, device)
    
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