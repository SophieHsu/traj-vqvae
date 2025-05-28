
from models.vqvae import RNNVQVAE, GumbelCodebookFreeVAE
from models.teacher import MaskedMeanClassifier, FinalStepClassifier
from load_traj import TrajectoryLatentDataset, MultiAgentTrajectoryDataset, trajectory_latent_collate_fn
from human_knowledge_utils import misclassification_validity_check, BalancedAgentSampler, get_agent_to_indices

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import numpy as np
import os

from dataclasses import dataclass
import tyro
import wandb
import time
import random
from tqdm import tqdm
import yaml
import inspect
import h5py

TEACHER_MODELS = {
    "MaskedMeanClassifier": MaskedMeanClassifier,
    "FinalStepClassifier": FinalStepClassifier,
}

VAE_MODELS = {
    "RNNVQVAE": RNNVQVAE,
    "GumbelCodebookFreeVAE": GumbelCodebookFreeVAE,
}

@dataclass
class Args:
    """Dataset Setting"""
    model_dir: str = None # directory the model checkpoint and config yaml are stored
    checkpoint_file: str = None # name of vqvae checkpoint file inside model_dir
    raw_data_path: str = None # raw trajectory data path used to generate new dataset if data_path is not provided. ignored if data_path is provided
    vae_type: str = None
    data_path: str = None
    balanced_sampling: bool = True

    """Torch, cuda, seed"""
    exp_name: str = "masked_mean_classifier"
    seed: int = 1 # NOTE 
    torch_deterministic: bool = True
    cuda: bool = True
    
    """Logging"""
    track: bool = False
    wandb_project_name: str = "human-knowledge-teacher"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    save_interval: int = 100 # number of epochs between each checkpoint saving

    """Teacher Model Settings"""
    teacher_model_type: str = "MaskedMeanClassifier"
    hidden_dim: int = 256

    """Training Hyperparams"""
    num_epochs: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-4


# def prepare_data(raw_data_path=None, model_path=None, model_config_path=None, data_path=None):
def prepare_data(raw_data_path=None, model_dir=None, checkpoint_file=None, data_path=None):
    """
    Generate training and validation dataset from either one of the below
        1. teacher training dataset hdf5 file path
        2. raw trajectory dataset hdf5 file path AND trained VQVAE model path AND model config path

    Args:
        raw_data_path (optional): path to raw trajectory dataset hdf5 path
        model_dir (optional): path to directory containing trained VQVAE model and config files
        checkpoint_file (optional): name of the VQVAE model checkpoint file inside model_dir
        data_path (optional): path to teacher training dataset hdf5 path
    """
    # if only the raw data is provided, generate the data and save it
    if data_path is None:
        print("Generating new dataset")
        assert raw_data_path is not None and model_dir is not None and checkpoint_file is not None
        print("Loading VQVAE model...")
        # load model
        model_config_path = os.path.join(model_dir, "files", "config.yaml")
        model_path = os.path.join(model_dir, "files", checkpoint_file)
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.SafeLoader)
        # get all model arguments
        vae_class = VAE_MODELS[args.vae_type]
        model_input_keys = list(inspect.signature(vae_class.__init__).parameters.keys())
        model_input_keys.remove('self')
        model_kwargs = {key: model_config[key]["value"] for key in model_input_keys if key in model_config}
        model_kwargs["n_past_steps"] = model_config["input_seq_len"]["value"]
        vqvae_model = vae_class(**model_kwargs)
        vqvae_model.load_state_dict(torch.load(model_path, weights_only=True))
        print(vqvae_model)

        # load raw dataset 
        print("Loading raw dataset...")
        raw_dataset = MultiAgentTrajectoryDataset(
            data_file_path=raw_data_path, 
            sequence_len=model_config["input_seq_len"]["value"],
        )

        # encode raw data
        print("Generating teacher training dataset...")
        data_save_dir = os.path.join(
            f"data/teacher/minigrid",
            os.path.basename(os.path.dirname(raw_data_path)), # name of raw trajectory dataset
            os.path.basename(model_dir) # name of VQVAE model used to generate embeddings
        )
        os.makedirs(data_save_dir, exist_ok=True)
        data_path = os.path.join(data_save_dir, os.path.basename(raw_data_path))
        print(f"Data will be saved to {data_path}")
        teacher_data_file = h5py.File(data_path, "w")
        traj_group = teacher_data_file.create_group("embeddings")
        for idx, traj in enumerate(tqdm(raw_dataset)):
            # get data
            state = torch.tensor(traj["state0"], dtype=torch.float32)
            action_indices = torch.tensor(traj["action_indices"], dtype=torch.long)
            reward = torch.tensor(traj["reward"], dtype=torch.float32)
            mask = torch.tensor(traj["mask"], dtype=torch.long)
            agent_id = traj["agent_id"]
            vis_collision_hist = torch.tensor(traj["vis_wall_collision_hist"], dtype=torch.bool)
            invis_collision_hist = torch.tensor(traj["invis_wall_collision_hist"], dtype=torch.bool)

            x = torch.cat([state, action_indices.unsqueeze(-1).float(), reward.unsqueeze(-1)], dim=-1)
            z_e, z_q, min_encodings, min_encoding_indices = vqvae_model.get_embeddings(
                x={
                    "traj": x.unsqueeze(0),
                    "mask": mask.unsqueeze(0),
                }, 
            )
            traj_i = traj_group.create_group(traj["traj_name"])
            traj_i.create_dataset("z_e", shape=z_e.shape[1:], dtype=float, data=z_e.detach().cpu())
            traj_i.create_dataset("z_q", shape=z_q.shape[1:], dtype=float, data=z_q.detach().cpu())
            traj_i.create_dataset("agent_id", shape=(1,), dtype=int, data=agent_id)
            traj_i.create_dataset("mask", shape=mask.shape, dtype=int, data=mask)
            traj_i.create_dataset("action", shape=action_indices.shape, dtype=int, data=action_indices)
            traj_i.create_dataset("vis_wall_collision_hist", shape=vis_collision_hist.shape, dtype=bool, data=vis_collision_hist)
            traj_i.create_dataset("invis_wall_collision_hist", shape=invis_collision_hist.shape, dtype=bool, data=invis_collision_hist)

        teacher_data_file.close()
        print("Done")
    
    # generate teacher training dataset from datafile
    full_dataset = TrajectoryLatentDataset(data_file_path=data_path)
    n_data = len(full_dataset)
    n_train = int(0.2 * n_data)
    n_val = n_data - n_train
    train_data, val_data = random_split(full_dataset, [n_train, n_val])
    
    return train_data, val_data

def train_teacher(model, train_loader, val_loader, num_epochs, learning_rate, device):

    # setup wandb logging
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
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

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    agent_names = [f"A{i}" for i in range(6)]
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0
        correct = 0
        train_total = 0
        train_preds, train_gts = [], []
        actions = []
        invis_collision_hist = []
        """
        TRAINING
        """
        sampled_agents = []
        for batch in train_loader:
            z_q = batch["z_q"].to(device)
            mask = batch["mask"].to(device)
            agent_id = batch["agent_id"].to(device)
            actions.append(batch["action"])
            invis_collision_hist.append(batch["invis_wall_collision_hist"])

            logits = model(z_q, mask)
            loss = F.cross_entropy(logits, agent_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * z_q.shape[0]
            preds = logits.argmax(dim=1)
            train_preds.append(preds.detach().cpu())
            train_gts.append(agent_id.detach().cpu())
            correct += (preds == agent_id).sum().item()
            train_total += agent_id.shape[0]

            sampled_agents.append(agent_id)

        train_avg_loss = total_train_loss / train_total
        train_accuracy = correct / train_total 
        # print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_avg_loss:.4f} | Acc: {train_accuracy:.4f}%")
        train_misclass_validity_results = misclassification_validity_check(
            preds=torch.cat(train_preds),
            true_labels=torch.cat(train_gts),
            action_trajs=torch.cat(actions, dim=0),
            n_invisible_collisions=torch.cat(invis_collision_hist).sum(dim=1),
        )

        """
        VALIDATION
        """
        model.eval()
        val_correct = 0
        total_val_loss = 0
        val_total = 0
        valid_preds, valid_gts = [], []
        actions = []
        invis_collision_hist = []
        with torch.no_grad():
            for batch in val_loader:
                z_q = batch["z_q"].to(device)
                mask = batch["mask"].to(device)
                agent_id = batch["agent_id"].to(device)
                actions.append(batch["action"])
                invis_collision_hist.append(batch["invis_wall_collision_hist"])

                logits = model(z_q, mask)
                loss = F.cross_entropy(logits, agent_id)

                total_val_loss += loss.item() * z_q.size(0)
                preds = logits.argmax(dim=1)
                valid_preds.append(preds.detach().cpu())
                valid_gts.append(agent_id.detach().cpu())
                val_correct += (preds == agent_id).sum().item()
                val_total += agent_id.size(0)

        val_avg_loss = total_val_loss / val_total
        val_accuracy = val_correct / val_total 

        val_misclass_validity_results = misclassification_validity_check(
            preds=torch.cat(valid_preds),
            true_labels=torch.cat(valid_gts),
            action_trajs=torch.cat(actions, dim=0),
            n_invisible_collisions=torch.cat(invis_collision_hist).sum(dim=1),
        )
        if args.track:
            wandb.run.log({
                "train_loss": train_avg_loss,
                "valid_loss": val_avg_loss,

                "train_accuracy": train_accuracy,
                "valid_accuracy": val_accuracy,

                "train_confusion": wandb.plot.confusion_matrix(
                    preds=torch.cat(train_preds).tolist(),
                    y_true=torch.cat(train_gts).tolist(),
                    class_names=agent_names,
                    title="train agent classification",
                ),
                "valid_confusion": wandb.plot.confusion_matrix(
                    preds=torch.cat(valid_preds).tolist(),
                    y_true=torch.cat(valid_gts).tolist(),
                    class_names=agent_names,
                    title="valid agent classificaiton",
                ),
                
                "train_valid_incorrect_per_incorrect": train_misclass_validity_results["n_valid_incorrect"] / train_misclass_validity_results["n_incorrect"],
                "train_valid_incorrect_per_total": train_misclass_validity_results["n_valid_incorrect"] / train_misclass_validity_results["n_total"],

                "valid_valid_incorrect_per_incorrect": val_misclass_validity_results["n_valid_incorrect"] / val_misclass_validity_results["n_incorrect"],
                "valid_valid_incorrect_per_total": val_misclass_validity_results["n_valid_incorrect"] / val_misclass_validity_results["n_total"],
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



def main(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # initialize dataloaders
    train_data, val_data = prepare_data(
        raw_data_path=args.raw_data_path,
        model_dir=args.model_dir,
        checkpoint_file=args.checkpoint_file,
        data_path=args.data_path,
    )
    if args.balanced_sampling:
        train_agent_to_data_idx = get_agent_to_indices(train_data)
        valid_agent_to_data_idx = get_agent_to_indices(val_data)
        train_sampler = BalancedAgentSampler(agent_to_indices=train_agent_to_data_idx)
        valid_sampler = BalancedAgentSampler(agent_to_indices=valid_agent_to_data_idx)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            pin_memory=True,
            collate_fn=trajectory_latent_collate_fn,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            pin_memory=True,
            collate_fn=trajectory_latent_collate_fn,
            sampler=valid_sampler,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=trajectory_latent_collate_fn,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=trajectory_latent_collate_fn,
        )
    print("Dataloaders ready")

    # initialize model
    model_class = TEACHER_MODELS[args.teacher_model_type]
    model = model_class(
        embedding_dim=train_data[0]["z_q"].shape[1],
        num_classes=6,
        hidden_dim=args.hidden_dim,
    )
    print("Teacher model initialized")
    print(model)

    # train
    train_teacher(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)