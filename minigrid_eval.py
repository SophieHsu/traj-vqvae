import argparse
import torch

from models.vqvae import VQVAE


def main(args):
    model_data = torch.load(args.model, weights_only=False) # contain state_dict, results (losses), hyperparameters
    params = model_data["hyperparameters"]
    model = VQVAE(
        h_dim=params["h_dim"],
        res_h_dim=params["res_h_dim"],
        n_res_layers=params["n_res_layers"],
        n_embeddings=params["n_embeddings"],
        embedding_dim=params["embedding_dim"],
        beta=params["beta"],
    )
    model.load_state_dict(model_data["model"])
    model.eval()

    # create dataloader
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        help="vq-vae model to evaluate",
        default="/home/ayanoh/traj-vqvae/results/vqvae_data_fri_apr_4_14_36_07_2025.pth",
    )
    parser.add_argument(
        "--datafile", 
        type=str, 
        help="path to evaluation dataset",
        default="/home/ayanoh/traj-vqvae/data/minigrid/4-rooms-1k/combined.hdf5",
    )
    
    args = parser.parse_args()
    main(args)