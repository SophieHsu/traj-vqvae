
from models.vqvae import RNNVQVAE, GumbelCodebookFreeVAE
from load_traj import TrajectoryLatentDataset, MultiAgentTrajectoryDataset, trajectory_latent_collate_fn
from human_knowledge_utils import BalancedAgentSampler
from datasets.prepare_minigrid_dataset import AGENTS, get_agent_id

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import gymnasium as gym
import minigrid
from minigrid.core.actions import create_custom_action_enum

import numpy as np
import os
from enum import IntEnum

from dataclasses import dataclass
from typing import Tuple, Optional
import tyro
import wandb
import time
import random
from tqdm import tqdm
import yaml
import inspect
import h5py

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
    exp_name: str = "teacher_unlock_test"
    seed: int = 1 # NOTE 
    torch_deterministic: bool = True
    cuda: bool = True
    
    """Logging"""
    track: bool = False
    wandb_project_name: str = "human-knowledge-teacher-unlock"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    save_interval: int = 100 # number of epochs between each checkpoint saving

    """Student Settings"""
    student_model_dir: str = "/home/ayanoh/human-knowledge/proxy_models/9-rooms"

    """VQVAE Model Settings"""
    vqvae_model_dir: str = "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-binary-30p-10f-balanced"
    vqvae_checkpoint_file: str = "checkpoint_epoch_1999.pt"

    """Teacher Model Settings"""
    hidden_dim: int = 256
    n_student_types: int = 6
    n_teacher_actions: int = 5
    lstm_size: int = 128
    use_belief_head: bool = True

    """Training Hyperparams"""
    num_epochs: int = 5000
    n_rollout_steps_for_voi_computation: int = 4 # in current setup, agent can move at least one grid in 4 steps

    """Algorithm specific arguments"""
    total_timesteps: int = int(1e9)
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 128 # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95 # the lambda for the general advantage estimation
    num_minibatches: int = 1
    update_epochs: int = 4 # the K epochs to update the policy
    norm_adv: bool = True # Toggles advantages normalization
    clip_coef: float = 0.1 # the surrogate clipping coefficient
    clip_vloss: bool = True # Toggles whether or not to use a clipped loss for the value function, as per the paper
    ent_coef: float = 0.01 # coefficient of the entropy
    vf_coef: float = 0.5 # coefficient of the value function
    max_grad_norm: float = 0.5 # the maximum norm for the gradient clipping
    target_kl: Optional[float] = None # the target KL divergence threshold

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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

class TeacherAgent(nn.Module):
    def __init__(
        self, 
        zq_dim, 
        state_dim, 
        action_dim, 
        hidden_dim,
        n_student_types,
        lstm_size=128, 
        belief_head=True,
    ):
        super().__init__()
        self.input_dim = zq_dim + state_dim + 1 # +1 for teacher's action history
        self.lstm_size = lstm_size
        self.hidden_dim = hidden_dim
        self.use_belief_head = belief_head
        self.n_student_types = n_student_types
        self.action_dim = action_dim

        self.network = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.lstm_size)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.lstm_size, self.hidden_dim)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)

        # belief head
        if self.use_belief_head:
            self.belief_head = layer_init(nn.Linear(self.hidden_dim, self.n_student_types))

        # action head (takes hidden state and belief if used)
        policy_input_dim = self.hidden_dim + (self.n_student_types if self.use_belief_head else 0)
        self.actor = layer_init(nn.Linear(policy_input_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(policy_input_dim, 1), std=1)

    def get_states(self, x, lstm_state, done):
        """
        x: shape (B, input_dim)
        done: shape (B,)
        lstm_state: tuple of (h, c) with shape (1, B, hidden_dim)
        """
        hidden = self.network(x)  # (B, lstm_size)
        hidden = hidden.unsqueeze(0)  # (1, B, lstm_size)
        done = done.view(1, -1, 1)  # (1, B, 1)

        # Reset hidden state if done
        h_in = (1.0 - done) * lstm_state[0]
        c_in = (1.0 - done) * lstm_state[1]
        out, lstm_state = self.lstm(hidden, (h_in, c_in))
        out = out.squeeze(0)  # (B, hidden_dim)
        return out, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        if self.use_belief_head:
            belief_logits = self.belief_head(hidden)
            belief_dist = torch.softmax(belief_logits, dim=-1)
            hidden = torch.cat([hidden, belief_dist], dim=-1)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        if self.use_belief_head:
            belief_logits = self.belief_head(hidden)
            belief_dist = torch.softmax(belief_logits, dim=-1)
            hidden = torch.cat([hidden, belief_dist], dim=-1)

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
    
    def get_belief(self, x, lstm_state, done):
        assert self.use_belief_head, "belief head does not exist"
        hidden, _ = self.get_states(x, lstm_state, done)
        belief_logits = self.belief_head(hidden)
        belief_dist = torch.softmax(belief_logits, dim=-1)
        return belief_dist

def load_student_configs(student_model_dir):
    """
    Returns a dictionary of agent ID to model checkpoint paths
    """
    checkpoints = {}
    configs = {}
    obs_types = {}
    student_knowledges = {}
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
        elif "nored" in file:
            obs_types[agent_id] = "nored"
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
            
    return checkpoints, configs, obs_types, student_knowledges

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
        unexpected_state_penalty, n_frame_stacks,
        update_state, shaped_reward, reward_modes,
    ):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                env_id,
                max_steps=max_steps,
                action_type=action_type,
                render_mode="rgb_array",
                unexpected_state_penalty=unexpected_state_penalty,
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
                unexpected_state_penalty=unexpected_state_penalty,
                n_frame_stacks=n_frame_stacks,
                update_state=update_state,
                shaped_reward=shaped_reward,
                reward_modes=reward_modes,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def take_step(agent, envs, action_enum, obs_type, next_obs, next_done, next_lstm_state, device):
    """
    Makes a specified agent take a step in the environment
    Re-maps agent actions to environment action indices using action_enum
    
    Args:
        action_enum (IntEnum) : maps agent's actions to action indices

    TODO return
    should return lstm_states to pass it on to the next agent
    """
    assert len(envs.envs) == 1, "Running multiple environments is currently not supported. Check TODOs (non-exhaustive) in code."

    # get action from student policy
    actions, _, _, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
    
    # remap actions
    env_action_enum = envs.envs[0].unwrapped.actions
    action_names = [action_enum(action.item()).name for action in actions]
    remapped_actions = torch.tensor([env_action_enum[action_name].value for action_name in action_names]).to(device)

    # set observation type
    envs.envs[0].unwrapped.set_obs_type(obs_type=obs_type) # TODO - needs change if running multiple environments

    next_obs, reward, terminations, truncations, infos = envs.step(remapped_actions.cpu().numpy())
    next_done = np.logical_or(terminations, truncations)
    next_obs, next_done, reward = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device), torch.Tensor(reward).to(device)

    return next_obs, remapped_actions, reward, next_done, next_lstm_state, infos

def rollout_students_and_get_dist_to_goal(
    teacher_env_states,
    rollout_envs, 
    student_agents,
    student_action_enums,
    student_obs_types, 
    initial_student_lstm_state,
    n_rollout_steps, 
    device,
):
    # NOTE / TODO - assumes (1) the episode has not terminated when this function is called (2) there is one environment
    
    rollout_envs.reset()
    """
    Rolls out student agents in the environment for n_rollout_steps steps
    """
    n_students = len(student_agents)
    n_steps_taken = np.zeros(n_students, dtype=int)
    done = torch.zeros(n_students).to(device)
    lstm_states = [initial_student_lstm_state for _ in range(n_students)]

    # Restore it into all rollout environments
    for i, env in enumerate(rollout_envs.envs):
        env.unwrapped.load_internal_state(teacher_env_states)
        env.unwrapped.set_obs_type(student_obs_types[i])  # important for student perspective

    obs = np.stack([env.unwrapped.gen_obs() for env in rollout_envs.envs])
    obs = torch.tensor(obs, dtype=torch.float32).to(device)

    for step in range(n_rollout_steps):
        actions = []
        for i in range(n_students):
            student = student_agents[i]
            action, _, _, _, lstm_state = student.get_action_and_value(
                obs[i].unsqueeze(0), lstm_states[i], done[i].unsqueeze(0))
            lstm_states[i] = lstm_state
            action_enum = student_action_enums[i]
            env_action_enum = rollout_envs.envs[i].unwrapped.actions
            action_name = action_enum(action.item()).name
            actions.append(env_action_enum[action_name].value)
            n_steps_taken[i] += 1

        obs_, _, terminations, truncations, _ = rollout_envs.step(np.array(actions))
        done = torch.tensor(np.logical_or(terminations, truncations), dtype=torch.float32).to(device)
        obs = torch.tensor(obs_, dtype=torch.float32).to(device)
        if done.all():
            break

    # compute final distances
    dists = np.array([env.unwrapped.compute_n_grids_to_goal() for env in rollout_envs.envs])
    return dict(enumerate(dists)), dict(enumerate(n_steps_taken))

def compute_voi_reward(
    student_knowledges, 
    student_knowledge_to_id, 
    utilities, 
    teacher_action, 
    student_type_belief,
):
    """
    TODO - probably should be vectorized

    Compute VoI reward as:
        sum_S( P(S) * P(S'|S,a_T) * U(S') ) - sum_S( P(S) * U(S) )
        (see notion 06/05/2025)
    Args:
        student_knowledges (dict) : map student ID to knowledge
        student_knowledge_to_id (dict) : map student knowledge (tuple) to ID
        utilities (dict) : utility of each state (where state is student ID)
        student_type_belief (array) : belief over each student type
    """
    value_after_action, value_before_action = 0, 0

    for student_id in student_knowledges.keys():
        # find the student_id after teacher action
        new_student_id = get_new_student_id(student_id, teacher_action, student_knowledges, student_knowledge_to_id)
        
        # get value after teacher action
        value_after_action += student_type_belief[student_id] * utilities[new_student_id]

        # get value before teacher action
        value_before_action += student_type_belief[student_id] * utilities[student_id]
    
    return value_after_action - value_before_action

def get_new_student_id(student_id, teacher_action, student_knowledges, student_knowledge_to_id):
    """
    get student ID after teacher action
    """
    new_knowledge = student_knowledges[student_id].copy()
    if teacher_action.item() == TeacherActions.do_nothing:
        # the last teacher action corresponds to "do nothing" action
        return student_id
    else:
        new_knowledge[teacher_action.item()] = 1
        new_student_id = student_knowledge_to_id[tuple(new_knowledge)]
        return new_student_id

def main(args):

    # torch.backends.cudnn.enabled = False

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
    student_checkpoints, student_configs, student_obs_types, student_knowledges = load_student_configs(args.student_model_dir)
    student_knowledge_to_id = {tuple(val): key for key, val in student_knowledges.items()}

    # get environment configs
    env_configs = get_environment_configs(student_configs)

    """
    Setup Environment
    """
    # env setup
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(
    #         args.env_id, i, args.capture_video, run_name, 
    #         max_steps, action_type, 
    #         unexpected_state_penalty, n_frame_stacks,
    #         update_state, shaped_reward, reward_modes,
    #     ) for i in range(args.num_envs)],
    # )
    # 1 env for now. TODO - is there way to parallelize multiple envs?
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            env_configs["env_id"], 
            0, 
            env_configs["capture_video"], 
            run_name, 
            env_configs["max_steps"],
            env_configs["action_type"], 
            env_configs["unexpected_state_penalty"], 
            env_configs["n_frame_stacks"],
            env_configs["update_state"], 
            env_configs["shaped_reward"], 
            env_configs["reward_modes"],
        )],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    """
    Initialize Student Models
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
    Create environments used for student rollouts during VoI computation
    """
    rollout_envs = gym.vector.SyncVectorEnv([
        make_env(
            env_configs["env_id"],
            idx=i,
            capture_video=False,
            run_name="rollout_env",
            max_steps=env_configs["max_steps"],
            action_type=env_configs["action_type"],
            unexpected_state_penalty=env_configs["unexpected_state_penalty"],
            n_frame_stacks=env_configs["n_frame_stacks"],
            update_state=env_configs["update_state"],
            shaped_reward=env_configs["shaped_reward"],
            reward_modes=env_configs["reward_modes"],
        ) for i in range(len(student_models))
    ])
    rollout_envs.reset(seed=args.seed)


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
    Initialize Teacher Model and Optimizer
    """
    teacher_agent = TeacherAgent(
        zq_dim=zq_dim,
        state_dim=state_dim,
        action_dim=args.n_teacher_actions,
        hidden_dim=args.hidden_dim,
        n_student_types=args.n_student_types,
        lstm_size=args.lstm_size,
        belief_head=args.use_belief_head,
    )
    teacher_agent.to(device)
    optimizer = optim.Adam(teacher_agent.parameters(), lr=args.learning_rate, eps=1e-5)


    """
    Storage Setup
    """
    teacher_obs_buffer = torch.zeros((args.num_steps, teacher_agent.input_dim)).to(device)
    teacher_action_buffer = torch.zeros((args.num_steps), dtype=torch.long).to(device)
    student_traj_buffer = torch.zeros((args.num_steps, vqvae_model.in_dim)).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps)).to(device)
    # student_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    student_rewards = torch.zeros((args.num_steps)).to(device)
    teacher_rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros((args.num_steps)).to(device)
    values = torch.zeros((args.num_steps)).to(device)

    # Choose a random student
    student_id = np.random.choice(n_students)
    student_agent = student_models[student_id]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    envs.reset(seed=args.seed)
    envs.envs[0].unwrapped.set_obs_type(student_obs_types[student_id]) # set observation return type
    next_student_obs = envs.envs[0].unwrapped.gen_obs() # this is the observation student sees (may be different from what teacher sees)
    next_student_obs = torch.tensor(next_student_obs, dtype=torch.float32).to(device)
    next_teacher_obs = envs.envs[0].unwrapped.gen_gt_obs()
    next_teacher_obs = torch.tensor(next_teacher_obs, dtype=torch.float32).to(device)
    # next_obs, _ = envs.reset(seed=args.seed) # This is the student's observation
    # next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_student_lstm_state = (
        torch.zeros(student_agent.lstm.num_layers, args.num_envs, student_agent.lstm.hidden_size).to(device),
        torch.zeros(student_agent.lstm.num_layers, args.num_envs, student_agent.lstm.hidden_size).to(device),
    )
    next_teacher_lstm_state = (
        torch.zeros(teacher_agent.lstm.num_layers, args.num_envs, teacher_agent.lstm.hidden_size).to(device),
        torch.zeros(teacher_agent.lstm.num_layers, args.num_envs, teacher_agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    """
    Start main training loop
    """
    for iteration in tqdm(range(1, args.num_iterations + 1)):

        # clear out student trajectory buffer to avoid values from previous iterations affecting zq computation
        student_traj_buffer = torch.zeros((args.num_steps, vqvae_model.in_dim)).to(device)

        # initial_teacher_lstm_state = (next_teacher_lstm_state[0].clone(), next_teacher_lstm_state[1].clone())
        initial_teacher_lstm_state = (
            next_teacher_lstm_state[0].clone().detach(), 
            next_teacher_lstm_state[1].clone().detach(),
        )

        teacher_action = TeacherActions.do_nothing 

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # keep track of teacher performance per episode
        episode_teacher_reward = 0
        episode_n_redundant_disclosures = 0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            """
            Step the current student
            """
            next_student_obs, student_actions, student_reward, next_done, next_student_lstm_state, student_infos = take_step(
                agent=student_agent,
                envs=envs,
                action_enum=student_action_enums[student_id],
                obs_type=student_obs_types[student_id],
                next_obs=next_student_obs,
                next_done=next_done,
                next_lstm_state=next_student_lstm_state,
                device=device,
            )
            dones[step] = next_done.item()
            student_rewards[step] = student_reward.item()

            """
            Get teacher observation (zq, s_t, a_T)
            """
            # encode student trajectory into zq
            # concatenate teacher state, student action, and student reward into a single step of student trajectory, add to buffer
            traj_step = torch.cat([ # 1 step of student trajectory
                next_teacher_obs, 
                student_actions.float(), 
                student_reward,
            ])
            student_traj_buffer[step] = traj_step
            # 
            num_valid_steps = min(step + 1, n_vqvae_input_steps) # number of real (unpadded) steps to encode
            start_idx = max(0, step - n_vqvae_input_steps)
            mask = torch.zeros(n_vqvae_input_steps, dtype=torch.float32).to(device)
            mask[:num_valid_steps] = 1
            _, z_q, _, _ = vqvae_model.get_embeddings(
                x={
                    "traj": student_traj_buffer[start_idx:start_idx+n_vqvae_input_steps].unsqueeze(0),
                    "mask": mask.unsqueeze(0),
                }, 
            )

            # concat zq, st, teacher_action -> put in teacher obs buffer
            z_q = z_q.detach()
            zq_last_step = z_q[0,num_valid_steps-1,:].to(device) # NOTE - needs fixed if training multiple environments
            teacher_obs = torch.cat([zq_last_step, next_teacher_obs, torch.tensor([teacher_action]).to(device)]).detach()
            teacher_obs_buffer[step] = teacher_obs

            """
            Take teacher action
            """
            # ALGO LOGIC: action logic
            with torch.no_grad():
                teacher_action, logprob, _, value, next_teacher_lstm_state = teacher_agent.get_action_and_value(
                    teacher_obs_buffer[step].unsqueeze(0), 
                    next_teacher_lstm_state, 
                    next_done,
                )
                values[step] = value.flatten()
            teacher_action_buffer[step] = teacher_action
            logprobs[step] = logprob.item()

            # find out agent type after the teacher takes action
            new_student_id = get_new_student_id(student_id, teacher_action, student_knowledges, student_knowledge_to_id)

            """
            Compute VoI reward
            """
            # roll out the student before and after teacher action and get the distance to goal afterwards
            dist_to_goal, n_steps_completed = rollout_students_and_get_dist_to_goal(
                teacher_env_states=envs.envs[0].unwrapped.get_internal_state(),
                rollout_envs=rollout_envs,
                student_agents=student_models,
                student_action_enums=student_action_enums,
                student_obs_types=student_obs_types,
                initial_student_lstm_state=next_student_lstm_state,
                n_rollout_steps=args.n_rollout_steps_for_voi_computation,
                device=device,
            )
            utilities = {id : -dist_to_goal[id] + args.n_rollout_steps_for_voi_computation - n_steps_completed[id] for id in dist_to_goal.keys()}
            
            # get belief over student types
            with torch.no_grad():
                student_type_belief = teacher_agent.get_belief(
                    x=teacher_obs_buffer[step].unsqueeze(0),
                    lstm_state=next_teacher_lstm_state,
                    done=next_done,
                )[0].detach().cpu().numpy()
            
            reward_voi = compute_voi_reward(
                student_knowledges=student_knowledges,
                student_knowledge_to_id=student_knowledge_to_id,
                utilities=utilities,
                teacher_action=teacher_action,
                student_type_belief=student_type_belief,
            )
            teacher_rewards[step] = reward_voi.item()
            episode_teacher_reward += reward_voi.item()
            redundant = (
                teacher_action != TeacherActions.do_nothing and
                student_knowledges[student_id][teacher_action.item()] == 1
            )
            episode_n_redundant_disclosures += int(redundant)

            """
            Log teacher and student performances at the end of episode
            """
            if "final_info" in student_infos:
                final_student_info = student_infos["final_info"][0]
                episode_len = final_student_info["episode"]["l"]
                # student performance
                writer.add_scalar("charts/student_episodic_reward", final_student_info["episode"]["r"])
                writer.add_scalar("charts/student_episodic_length", final_student_info["episode"]["l"])
                # teacher performance
                writer.add_scalar("charts/episodic_voi_reward", episode_teacher_reward, global_step)
                writer.add_scalar("charts/episodic_redundant_rate", episode_n_redundant_disclosures / episode_len)
                writer.add_scalar("charts/episodic_redundant_count", episode_n_redundant_disclosures)

                episode_teacher_reward = 0
                episode_n_redundant_disclosures = 0

            student_id = new_student_id

        # bootstrap value if not done
        with torch.no_grad():
            next_value = teacher_agent.get_value(
                teacher_obs.unsqueeze(0),
                next_teacher_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(teacher_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = teacher_rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = teacher_obs_buffer.view(-1, teacher_obs_buffer.shape[-1])
        b_logprobs = logprobs.reshape(-1)
        b_actions = teacher_action_buffer.view(-1)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = teacher_agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_teacher_lstm_state[0][:, mbenvinds], initial_teacher_lstm_state[1][:, mbenvinds]),
                    # (initial_teacher_lstm_state[0][:, mbenvinds].detach(), initial_teacher_lstm_state[1][:, mbenvinds].detach()),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(teacher_agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and (iteration % args.save_interval == 0) and iteration > 0:
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(teacher_agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)))
        # print("SPS:", int(global_step / (time.time() - start_time)))

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)