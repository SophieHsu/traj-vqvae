
from models.vqvae import RNNVQVAE
from models.teacher import VoICriticTeacherModel
from datasets.prepare_minigrid_dataset import get_agent_id

import gymnasium as gym
import minigrid
from minigrid.core.actions import create_custom_action_enum

import yaml
import numpy as np
import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from enum import IntEnum

from dataclasses import dataclass
import wandb
import tyro
from tqdm import tqdm
import inspect


class Knowledge(IntEnum):
    """
    Possible student agent knowledge
    """
    obs_red_wall = 0
    act_left = 1
    act_right = 2
    act_forward = 3

class TeacherActions(IntEnum):
    """
    What teacher actions reveal
    """
    obs_red_wall = Knowledge.obs_red_wall
    act_left = Knowledge.act_left
    act_right = Knowledge.act_right
    act_forward = Knowledge.act_forward
    do_nothing = len(Knowledge)

@dataclass
class Args:
    """Torch, cuda, seed"""
    exp_name: str = "max_voi_teacher"
    seed: int = 1 # NOTE 
    torch_deterministic: bool = True
    cuda: bool = True
    
    """Logging"""
    track: bool = False
    wandb_project_name: str = "human-knowledge-teacher-unlock-value"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    save_interval: int = 100 # number of epochs between each checkpoint saving

    """Student Settings"""
    student_model_dir: str = "/home/ayanoh/human-knowledge/proxy_models/9-rooms"

    """VQVAE Model Settings"""
    vqvae_model_dir: str = "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-binary-30p-10f-balanced"
    vqvae_checkpoint_file: str = "checkpoint_epoch_1999.pt"

    """Training Hyperparams"""
    num_epochs: int = 5000

    """Teacher Settings"""
    n_student_types: int = 6
    n_teacher_actions: int = 5
    action_embedding_dim: int = 16
    teacher_hidden_dim: int = 128

    """Algorithm specific arguments"""
    total_timesteps: int = int(1e9)
    learning_rate: float = 2.5e-4
    num_envs: int = 2
    num_steps: int = 128 # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True
    num_minibatches: int = 1

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

OBS_TYPE_TO_VISIBLE_THINGS = {
    "all": ['grey', 'red', 'blue', 'purple'],
    "nored": ['grey', 'blue', 'purple'],
    "all-and-coins": ['grey', 'red', 'blue', 'purple', 'coins'],
    "nored-and-coins": ['grey', 'red', 'blue', 'purple', 'coins'],
}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StudentAgent(nn.Module):
    def __init__(self, envs, lstm_size, n_actions):
        super().__init__()
        self.lstm_size = lstm_size
        self.network = nn.Sequential(
            # layer_init(nn.Linear(86, 64)),
            layer_init(nn.Linear(envs.single_observation_space.shape[0], 128)),
            nn.ReLU(),
            # layer_init(nn.Linear(128, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.lstm_size)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.lstm_size, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
        self.actor = layer_init(nn.Linear(128, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

def load_student_configs(student_model_dir):
    """
    Returns a dictionary of agent ID to model checkpoint paths
    """
    checkpoints = {}
    configs = {}
    obs_types = {}
    student_knowledges = {}
    student_visible_things = {}
    for file in os.listdir(student_model_dir):
        agent_id = get_agent_id(file)
        checkpoints[agent_id] = os.path.join(student_model_dir, file, "files", "agent.pt")
        config_path = os.path.join(student_model_dir, file, "files", "config.yaml")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.load(f, Loader=yaml.SafeLoader)
        
        student_config = {}
        for key in raw_config.keys():

            student_config[key] = raw_config[key]["value"]
        configs[agent_id] = student_config

        student_knowlede = np.zeros(len(Knowledge))     
        # check observations
        if "all" in file:
            obs_types[agent_id] = "all"
            student_knowlede[Knowledge.obs_red_wall] = 1
            student_visibles = OBS_TYPE_TO_VISIBLE_THINGS["all"]
        elif "nored" in file:
            obs_types[agent_id] = "nored"
            student_visibles = OBS_TYPE_TO_VISIBLE_THINGS["nored"]
        else:
            raise NotImplementedError("only all and nored observation agents are supported at the moment")
        # check actions
        if "lrf" in file:
            student_knowlede[Knowledge.act_left] = 1
            student_knowlede[Knowledge.act_right] = 1
            student_knowlede[Knowledge.act_forward] = 1
        elif "lf" in file:
            student_knowlede[Knowledge.act_left] = 1
            student_knowlede[Knowledge.act_forward] = 1
        elif "rf" in file:
            student_knowlede[Knowledge.act_right] = 1
            student_knowlede[Knowledge.act_forward] = 1
        else:
            raise NotImplementedError()
        student_knowledges[agent_id] = student_knowlede
        student_visible_things[agent_id] = student_visibles
            
    return checkpoints, configs, obs_types, student_knowledges, student_visible_things

def get_environment_configs(student_configs):
    # Get env configs
    env_configs = {}
    env_configs["max_steps"] = max([cfg["max_steps"] for cfg in student_configs.values()])
    env_configs["action_type"] = max([cfg["action_type"] for cfg in student_configs.values()], key=len)
    env_configs["unexpected_state_penalty"] = student_configs[0]["unexpected_state_penalty"]
    env_configs["shaped_reward"] = student_configs[0]["shaped_reward"]
    env_configs["reward_modes"] = student_configs[0]["reward_modes"]
    env_configs["capture_video"] = student_configs[0]["capture_video"]
    # Confirm these configs are same for all students
    settings_to_check = ["n_frame_stacks", "update_state", "env_id"]
    for env_setting in settings_to_check:
        settings = [cfg[env_setting] for cfg in student_configs.values()]
        if env_setting == "env_id":
            # for env id, the environment has to be the same but observations can be different
            settings = [env_id.replace("nored", "*").replace("all", "*") for env_id in settings]
            assert all(i == settings[0] for i in settings), f"{env_setting} must be the same for all student agents"
            settings[0] = settings[0].replace("*", "all")
            env_configs[env_setting] = settings[0]
        else:
            assert all(i == settings[0] for i in settings), f"{env_setting} must be the same for all student agents"
            env_configs[env_setting] = settings[0]
    return env_configs

def make_env(
        env_id, idx, capture_video, run_name,
        max_steps, action_type,
        n_frame_stacks,
        update_state, shaped_reward, reward_modes,
    ):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                env_id,
                max_steps=max_steps,
                action_type=action_type,
                render_mode="rgb_array",
                n_frame_stacks=n_frame_stacks,
                update_state=update_state,
                shaped_reward=shaped_reward,
                reward_modes=reward_modes,
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                env_id,
                max_steps=max_steps,
                action_type=action_type,
                render_mode="rgb_array",
                n_frame_stacks=n_frame_stacks,
                update_state=update_state,
                shaped_reward=shaped_reward,
                reward_modes=reward_modes,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def take_step(agents, envs, action_enums, obs_types, next_obs, next_done, next_lstm_states, device):
    """
    Makes a specified agent take a step in the environment
    Re-maps agent actions to environment action indices using action_enum
    
    Args:
        action_enum (IntEnum) : maps agent's actions to action indices
    """

    num_envs = len(agents)
    remapped_actions = torch.zeros(num_envs, dtype=torch.long, device=device)
    new_lstm_h = []
    new_lstm_c = []

    for i in range(len(agents)):
        agent = agents[i]
        env = envs.envs[i]
        obs_type = obs_types[i]
        action_enum = action_enums[i]

        lstm_state_i = (next_lstm_states[0][:, i:i+1, :], next_lstm_states[1][:, i:i+1, :])
        
        # set observation type
        env.unwrapped.set_obs_type(visible_things=obs_type)
        
        # get action from student policy
        obs_i = next_obs[i].unsqueeze(0)  # (1, obs_dim)
        done_i = next_done[i].unsqueeze(0)  # (1,)

        with torch.no_grad():
            action, _, _, _, next_lstm_i = agent.get_action_and_value(obs_i, lstm_state_i, done_i)
        next_lstm_i = (next_lstm_i[0].detach(), next_lstm_i[1].detach())
        
        # remap actions
        agent_action_name = action_enum(action.item()).name
        remapped_action = env.unwrapped.actions[agent_action_name].value
        remapped_actions[i] = remapped_action

        # store updated LSTM state
        new_lstm_h.append(next_lstm_i[0])  # shape: (num_layers, 1, hidden_dim)
        new_lstm_c.append(next_lstm_i[1])

    # stack LSTM states along batch (env) dimension
    new_lstm_h = torch.cat(new_lstm_h, dim=1)  # (num_layers, num_envs, hidden_dim)
    new_lstm_c = torch.cat(new_lstm_c, dim=1)  # same
    next_lstm_states = (new_lstm_h, new_lstm_c)

    # step all envs together
    next_obs_np, reward_np, terminations, truncations, infos = envs.step(remapped_actions.cpu().numpy())
    next_done_np = np.logical_or(terminations, truncations)

    # convert to torch
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
    next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

    return next_obs, remapped_actions, reward, next_done, next_lstm_states, infos

def get_new_student_id(student_ids, teacher_action, student_knowledge_array):
    """
    get student ID after teacher action
    """
    new_knowledges = student_knowledge_array[student_ids].copy()
    ids_to_modify = np.nonzero(teacher_action.cpu() != TeacherActions.do_nothing).flatten()
    new_knowledges[ids_to_modify, teacher_action[ids_to_modify].cpu()] = 1
    # Compare each row in new_knowledges to all rows in student_knowledge_array
    matches = (new_knowledges[:, None, :] == student_knowledge_array[None, :, :]).all(axis=-1)

    # Find the index of the first matching row in student_knowledge_array for each new_knowledges row
    new_student_ids = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
    return new_student_ids

def compute_gt_voi(student_ids, student_values, num_teacher_actions, student_knowledge_array):
    """
    Compute ground truth VoI given by |U(S') - U(S)|
    """
    num_envs = len(student_ids)
    voi = np.zeros((num_envs, num_teacher_actions), dtype=np.float32)

    # Get current values V(S) for each env
    current_values = np.zeros((num_envs,), dtype=np.float32)
    for sid, v in student_values.items():
        mask = (student_ids == sid)
        current_values[mask] = v[mask].squeeze()

    # For each teacher action
    for a in range(num_teacher_actions):
        # Build a dummy tensor of teacher actions of shape (num_envs,)
        teacher_action_tensor = torch.full((num_envs,), a)

        # Compute the new student type after applying teacher action
        new_student_ids = get_new_student_id(student_ids, teacher_action_tensor, student_knowledge_array)

        # Get value estimates V(S') for new student types
        new_values = np.zeros((num_envs,), dtype=np.float32)
        for sid, v in student_values.items():
            mask = (new_student_ids == sid)
            new_values[mask] = v[mask].squeeze()

        # VoI = V(S') - V(S)
        voi[:, a] = abs(new_values - current_values)
    return voi

def main(args):
    assert args.n_teacher_actions == len(TeacherActions), "n_teacher_actions must match the number of knowledges for now"

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    """
    Load Configs from Students
    """
    # get student checkpoint and config paths
    student_checkpoints, student_configs, student_obs_types, student_knowledges, student_visible_things = load_student_configs(args.student_model_dir)
    student_knowledge_to_id = {tuple(val): key for key, val in student_knowledges.items()}
    all_student_ids = list(student_knowledges.keys())
    max_student_id = int(max(list(student_knowledges.keys())))
    student_knowledge_array = np.zeros((max_student_id+1, list(student_knowledges.values())[0].shape[0]))
    for sid in all_student_ids:
        student_knowledge_array[sid] = student_knowledges[sid]

    # get environment configs
    env_configs = get_environment_configs(student_configs)

    """
    Load knowledge representation model
    """
    # get checkpoint and config paths
    vqvae_checkpoint_path = os.path.join(args.vqvae_model_dir, "files", args.vqvae_checkpoint_file)
    vqvae_config_path = os.path.join(args.vqvae_model_dir, "files", "config.yaml")

    # extract model configs
    with open(vqvae_config_path, 'r') as f:
        vqvae_config = yaml.load(f, Loader=yaml.SafeLoader)
    vqvae_model_input_keys = list(inspect.signature(RNNVQVAE.__init__).parameters.keys())
    vqvae_model_input_keys.remove("self")
    vqvae_model_kwargs = {key: vqvae_config[key]["value"] for key in vqvae_model_input_keys if key in vqvae_config}
    vqvae_model_kwargs["n_past_steps"] = vqvae_config["input_seq_len"]["value"]

    # load model
    vqvae_model = RNNVQVAE(**vqvae_model_kwargs)
    vqvae_model.load_state_dict(torch.load(vqvae_checkpoint_path, weights_only=True))
    vqvae_model.to(device)
    vqvae_model.eval()
    state_dim = vqvae_model_kwargs["state_dim"]
    zq_dim = vqvae_model.embedding_dim
    n_vqvae_input_steps = vqvae_model.n_past_steps # number of input steps to the vqvae
    print(vqvae_model)
    print("#"*50, " Done loading representation VQVAE model ", "#"*50)

    """
    Setup Environment
    """
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            # env_configs["env_id"], 
            "MiniGrid-ColorObstacleNineRooms-test-v0",
            i, 
            env_configs["capture_video"], 
            run_name, 
            env_configs["max_steps"],
            env_configs["action_type"], 
            env_configs["n_frame_stacks"],
            env_configs["update_state"], 
            env_configs["shaped_reward"], 
            env_configs["reward_modes"],
        ) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    """
    Initialize Student Models
    TODO - update to match student class
    """
    student_models = {}
    student_action_enums = {}
    for agent_id, config in student_configs.items():
        student = StudentAgent(envs=envs, lstm_size=config["lstm_size"], n_actions=len(config["action_type"]))
        student_ckpt = torch.load(student_checkpoints[agent_id])
        student.load_state_dict(student_ckpt)
        student.to(device)
        student.eval()
        student_models[agent_id] = student
        student_action_enums[agent_id] = create_custom_action_enum(config["action_type"])
    n_students = len(student_models.keys())
    print("#"*50, " Done loading sutdent models ", "#"*50)
    
    """
    Set Up Logging
    """
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        if wandb.run is not None:
            wandb.run.log_code(".")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )   

    """
    Initialize Teacher Model
    """
    teacher_model = VoICriticTeacherModel(
        z_dim=zq_dim,
        state_dim=state_dim,
        n_teacher_actions=args.n_teacher_actions,
        n_student_types=args.n_student_types,
        action_embedding_dim=args.action_embedding_dim,
        hidden_dim=args.teacher_hidden_dim,
    ).to(device)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=args.learning_rate, eps=1e-5)

    """
    Prepare for Training
    """
    # buffer setup
    student_traj_buffer = torch.zeros((args.num_envs, args.num_steps, vqvae_model.in_dim), device=device)
    teacher_action_buffer = torch.zeros((args.num_envs, args.num_steps), dtype=torch.long, device=device)
    prev_teacher_action_buffer = torch.zeros((args.num_envs, args.num_steps), dtype=torch.long, device=device)
    env_state_buffer = torch.zeros((args.num_envs, args.num_steps, state_dim), device=device)
    zq_buffer = torch.zeros((args.num_envs, args.num_steps, zq_dim), device=device)
    student_type_labels_buffer = torch.zeros((args.num_envs, args.num_steps), dtype=torch.long, device=device)
    gt_voi_buffer = torch.zeros((args.num_envs, args.num_steps, args.n_teacher_actions), device=device)
    student_rewards_buffer = torch.zeros((args.num_envs, args.num_steps), device=device)
    dones_buffer = torch.zeros((args.num_envs, args.num_steps), device=device)

    # Choose a random student for each environment
    student_ids = np.random.choice(n_students, size=args.num_envs)
    student_agents = [student_models[sid] for sid in student_ids]
    # student_obs_type_list = [student_obs_types[sid] for sid in student_ids]
    student_visibles_list = [student_visible_things[sid] for sid in student_ids]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    envs.reset(seed=args.seed)

    # set right observation type for each environment
    for i in range(args.num_envs):
        envs.envs[i].unwrapped.set_obs_type(student_visible_things[student_ids[i]])

    # get initial student observations, dones, lestm states, gt_env_observations
    next_student_obs = torch.stack([
        torch.tensor(envs.envs[i].unwrapped.gen_obs(), dtype=torch.float32)
        for i in range(args.num_envs)
    ], dim=0).to(device)
    next_env_states = torch.stack([
        torch.tensor(envs.envs[i].unwrapped.gen_gt_obs(), dtype=torch.float32)
        for i in range(args.num_envs)
    ], dim=0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_student_lstm_state = (
        torch.zeros(student_agents[0].lstm.num_layers, args.num_envs, student_agents[0].lstm.hidden_size, device=device),
        torch.zeros(student_agents[0].lstm.num_layers, args.num_envs, student_agents[0].lstm.hidden_size, device=device),
    )

    """
    Main Training Loop
    """
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # clear buffers
        student_traj_buffer.zero_()
        teacher_action_buffer.zero_()
        prev_teacher_action_buffer.zero_()
        env_state_buffer.zero_()
        zq_buffer.zero_()
        student_type_labels_buffer.zero_()
        student_rewards_buffer.zero_()
        dones_buffer.zero_()
        
        # initial teacher action is do_nothing
        prev_teacher_action = torch.full((args.num_envs,), TeacherActions.do_nothing, device=device, dtype=torch.long)

        # keep track of teacher performance per episode
        episode_n_redundant_disclosures = torch.zeros(args.num_envs, device=device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        """
        Rollout
        """
        for step in range(0, args.num_steps):
            global_step += args.num_envs

            """
            Step the current student
            """
            next_student_obs, student_actions, student_reward, next_done, next_student_lstm_state, student_infos = take_step(
                agents=student_agents,
                envs=envs,
                action_enums=[student_action_enums[sid] for sid in student_ids],
                obs_types=student_visibles_list,
                next_obs=next_student_obs,
                next_done=next_done,
                next_lstm_states=next_student_lstm_state,
                device=device,
            )
            dones_buffer[:,step] = next_done.detach()
            student_rewards_buffer[:,step] = student_reward.detach()

            """
            Get input to VQVAE
            """
            # encode student trajectory into zq
            # concatenate teacher state, student action, and student reward into a single step of student trajectory, add to buffer
            traj_step = torch.cat([
                next_env_states, 
                student_actions.float().unsqueeze(1), 
                student_reward.unsqueeze(1),
            ], dim=1)
            student_traj_buffer[:,step] = traj_step.detach()

            # record the student type that just took this step
            student_type_labels_buffer[:, step] = torch.tensor(student_ids, dtype=torch.long, device=device)

            """
            Get embedding z_q
            """
            start_idx = max(0, step - n_vqvae_input_steps + 1)
            traj_chunk = student_traj_buffer[:, start_idx:step + 1]
            pad_len = n_vqvae_input_steps - traj_chunk.shape[1]
            padded_traj = F.pad(traj_chunk, (0, 0, 0, pad_len))
            mask = torch.zeros((args.num_envs, n_vqvae_input_steps), dtype=torch.float32, device=device)
            mask[:,:traj_chunk.shape[1]] = 1
            _, z_q, _, _ = vqvae_model.get_embeddings(
                x={
                    "traj": padded_traj,
                    "mask": mask,
                }
            )
            z_q = z_q.detach()
            zq_collapsed = z_q[:, traj_chunk.shape[1]-1] # TODO - currently last step. change to running mean
            zq_buffer[:,step] = zq_collapsed
            
            """
            Take teacher action
            """
            next_env_states = torch.stack([
                torch.tensor(envs.envs[i].unwrapped.gen_gt_obs(), dtype=torch.float32)
                for i in range(args.num_envs)
            ], dim=0).to(device)

            with torch.no_grad():
                teacher_output = teacher_model(zq_collapsed, prev_teacher_action, next_env_states)
            voi_preds = teacher_output["voi_preds"]
            # TODO - add cost of action
            teacher_action = torch.argmax(voi_preds, dim=1)
            teacher_action_buffer[:,step] = teacher_action.detach()
            prev_teacher_action_buffer[:,step] = prev_teacher_action.detach()
            
            # find out agent type after the teacher takes action
            new_student_ids = []
            new_student_ids = get_new_student_id(student_ids, teacher_action, student_knowledge_array)
            
            """
            Compute GT VoI
            """
            student_values = {}
    
            for sid in student_knowledges.keys():
                # stack observations from all environments
                student = student_models[sid]
                env_state_obs = np.stack([
                    env.unwrapped.gen_obs(visible_things=student_visible_things[sid]) 
                    for env in envs.envs
                ], axis=0)
                env_state_obs = torch.tensor(env_state_obs, dtype=torch.float32, device=device)
                # pass into critic to get values
                with torch.no_grad():
                    student_val = student.get_value(
                        env_state_obs, next_student_lstm_state, next_done
                    ).detach().cpu().numpy()
                    student_values[sid] = student_val
            # log critic outputs per student
            for sid, v in student_values.items():
                values_flat = v.flatten()
                writer.add_scalar(f"student_values/{sid}_mean", values_flat.mean(), global_step)
                writer.add_scalar(f"student_values/{sid}_std", values_flat.std(), global_step)
                writer.add_scalar(f"student_values/{sid}_min", values_flat.min(), global_step)
                writer.add_scalar(f"student_values/{sid}_max", values_flat.max(), global_step)

            voi_gts = compute_gt_voi(
                student_ids=student_ids,
                student_values=student_values,
                num_teacher_actions=args.n_teacher_actions,
                student_knowledge_array=student_knowledge_array,
            )
            gt_voi_buffer[:,step] = torch.tensor(voi_gts)

            voi_tensor = torch.tensor(voi_gts)  # shape (num_envs, n_teacher_actions)

            # log VoIs
            writer.add_scalar("voi_stats/mean", voi_tensor.mean().item(), global_step)
            writer.add_scalar("voi_stats/std", voi_tensor.std().item(), global_step)
            writer.add_scalar("voi_stats/min", voi_tensor.min().item(), global_step)
            writer.add_scalar("voi_stats/max", voi_tensor.max().item(), global_step)
            # per-action VoI mean
            for a in range(args.n_teacher_actions):
                writer.add_scalar(f"voi_stats/per_action/mean_action_{a}", voi_tensor[:, a].mean().item(), global_step)

            # count redundant teacher actions
            student_knowledge_before_action = student_knowledge_array[student_ids]
            do_nothing_mask = teacher_action.cpu() != TeacherActions.do_nothing 
            row_idx = np.arange(student_knowledge_before_action.shape[0])
            result = np.full((student_knowledge_before_action.shape[0],), float(0))
            result[do_nothing_mask] = student_knowledge_before_action[row_idx[do_nothing_mask], teacher_action[do_nothing_mask].cpu()]
            episode_n_redundant_disclosures += int(result.sum().item())
   
            """
            Log teacher and student performances at the end of episode
            """
            if "final_info" in student_infos:
                final_infos = student_infos["final_info"]
                finished_env_indices = [i for i, info in enumerate(final_infos) if info is not None]
                normalized_voi_rewards = []
                # NOTE - START HERE collect raw voi rewards for terminated environments and log the mean+++
                if len(finished_env_indices) > 0:
                    episode_lengths = np.array([final_infos[i]["episode"]["l"] for i in finished_env_indices])
                    episode_lengths = torch.tensor(episode_lengths, device=device)

                    for i in finished_env_indices:
                        writer.add_scalar("charts/student_episodic_reward", final_infos[i]["episode"]["r"], global_step + i)
                        writer.add_scalar("charts/student_episodic_length", final_infos[i]["episode"]["l"], global_step + i)

                    writer.add_scalar(
                        "charts/episodic_redundant_rate",
                        (episode_n_redundant_disclosures[finished_env_indices] / episode_lengths).mean().item(),
                        global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_redundant_count",
                        episode_n_redundant_disclosures[finished_env_indices].mean().item(),
                        global_step
                    )

                    # Reset only for finished environments
                    episode_n_redundant_disclosures[finished_env_indices] = 0

            student_ids = new_student_ids
            prev_teacher_action = teacher_action
            # print(f"alloc={torch.cuda.memory_allocated()/1e6:.1f} MB | "f"reserved={torch.cuda.memory_reserved()/1e6:.1f} MB")

        """
        Teacher Update
        """
        # flatten batch
        b_zq = zq_buffer.reshape(-1, zq_dim)
        b_prev_teacher_action = prev_teacher_action_buffer.reshape(-1)
        b_env_state = env_state_buffer.reshape(-1, state_dim)
        b_voi_targets = gt_voi_buffer.reshape(-1, args.n_teacher_actions)
        b_type_labels = student_type_labels_buffer.reshape(-1)

        # shuffle and minibatch
        indices = np.arange(b_zq.shape[0])
        np.random.shuffle(indices)

        student_type_pred_accuracies = []
        for start in range(0, len(indices), args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = indices[start:end]

            mb_zq = b_zq[mb_inds]
            mb_a_prev = b_prev_teacher_action[mb_inds]
            mb_env_state = b_env_state[mb_inds]
            mb_voi = b_voi_targets[mb_inds]
            mb_labels = b_type_labels[mb_inds]

            out = teacher_model(mb_zq, mb_a_prev, mb_env_state)
            pred_voi = out["voi_preds"]
            belief_logits = out["belief_logits"]

            # get student type prediction accuracy
            belief_preds = out["belief_probs"]
            pred_labels = belief_preds.argmax(dim=-1)
            accuracy = (pred_labels == mb_labels).float().mean()
            student_type_pred_accuracies.append(accuracy.item())

            # loss computation
            belief_loss = F.cross_entropy(belief_logits, mb_labels)
            voi_loss = F.mse_loss(pred_voi, mb_voi)
            loss = voi_loss + belief_loss

            # update
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()

        if args.track and (iteration % args.save_interval == 0) and iteration > 0:
            torch.save(teacher_optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(teacher_model.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
            
        writer.add_scalar("charts/learning_rate", teacher_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)))

        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/belief_loss", belief_loss.item(), global_step)
        writer.add_scalar("losses/voi_loss", voi_loss.item(), global_step)

        writer.add_scalar("metrics/student_type_pred_accuracy", np.mean(student_type_pred_accuracies), global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)