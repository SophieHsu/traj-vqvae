
from models.vqvae import RNNVQVAE
from models.teacher import VoICriticTeacherModel
from datasets.prepare_minigrid_dataset import get_agent_id

import gymnasium as gym
import minigrid
from minigrid.core.actions import create_custom_action_enum
from minigrid.utils.path_planning import get_astar_plan, get_dstar_plan

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
    wandb_project_name: str = "human-knowledge-teacher-unlock-planner"
    wandb_entity: str = "ahiranak-university-of-southern-california"
    save_interval: int = 100 # number of epochs between each checkpoint saving

    """VQVAE Model Settings"""
    vqvae_model_dir: str = "/home/ayanoh/traj-vqvae/trained_vqvae_models/9R2.5k-astar"
    vqvae_checkpoint_file: str = "checkpoint_epoch_200.pt"

    """Training Hyperparams"""
    num_epochs: int = 5000

    """Teacher Settings"""
    n_student_types: int = 6
    n_teacher_actions: int = 5
    action_embedding_dim: int = 16
    teacher_hidden_dim: int = 128
    n_steps_for_voi: int = 6
    z_flatten_method: str = "last_step"

    """Algorithm specific arguments"""
    total_timesteps: int = int(1e9)
    learning_rate: float = 2.5e-4
    num_envs: int = 4
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

### CAUTION - hardcoded dicts may require change if env is changed
STUDENT_KNOWLEDGES = { # red wall, left, right, forward
    0: np.array([1., 1., 1., 1.]),
    1: np.array([1., 1., 0., 1.]), 
    2: np.array([1., 0., 1., 1.]), 
    3: np.array([0., 1., 1., 1.]), 
    4: np.array([0., 1., 0., 1.]),
    5: np.array([0., 0., 1., 1.]), 
}
STUDENT_VISIBLE_THINGS = {
    0: OBS_TYPE_TO_VISIBLE_THINGS["all"],
    1: OBS_TYPE_TO_VISIBLE_THINGS["all"],
    2: OBS_TYPE_TO_VISIBLE_THINGS["all"],
    3: OBS_TYPE_TO_VISIBLE_THINGS["nored"],
    4: OBS_TYPE_TO_VISIBLE_THINGS["nored"],
    5: OBS_TYPE_TO_VISIBLE_THINGS["nored"],
}
STUDENT_ACTIONS = {
    0: ["left", "right", "forward"],
    1: ["left", "forward"],
    2: ["right", "forward"],
    3: ["left", "right", "forward"],
    4: ["left", "forward"],
    5: ["right", "forward"],
    # 0: [0, 1, 2],
    # 1: [0, 2],
    # 2: [1, 2],
    # 3: [0, 1, 2],
    # 4: [0, 2],
    # 5: [1, 2],
}

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

        # VoI = |V(S') - V(S)|
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
    Prepare students
    """
    student_knowledges = STUDENT_KNOWLEDGES
    n_students = len(student_knowledges)

    all_student_ids = list(student_knowledges.keys())
    max_student_id = int(max(list(student_knowledges.keys())))
    student_knowledge_array = np.zeros((max_student_id+1, list(student_knowledges.values())[0].shape[0]))
    for sid in all_student_ids:
        student_knowledge_array[sid] = student_knowledges[sid]

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
            env_id="MiniGrid-ColorObstacleNineRooms-test-v0",
            idx=i,
            capture_video=False,
            run_name=run_name,
            max_steps=400,
            action_type=["left", "right", "forward"],
            n_frame_stacks=1,
            update_state=True,
            shaped_reward=True,
            reward_modes=['reward_success', 'progress', 'exploration'],
        ) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

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

    steps_since_reset = torch.zeros((args.num_envs), device=device)


    # Choose a random student for each environment
    student_ids = np.random.choice(n_students, size=args.num_envs)
    student_visibles_list = [STUDENT_VISIBLE_THINGS[sid] for sid in student_ids]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    envs.reset(seed=args.seed)

    # set right observation type for each environment
    for i in range(args.num_envs):
        # envs.envs[i].unwrapped.set_obs_type(STUDENT_VISIBLE_THINGS[student_ids[i]])
        envs.envs[i].unwrapped.set_obs_type(student_visibles_list[i])

    # get initial student observations, dones, lestm states, gt_env_observations
    next_student_obs = torch.stack([
        torch.tensor(envs.envs[i].unwrapped.gen_obs(visible_things=student_visibles_list[i]), dtype=torch.float32)
        for i in range(args.num_envs)
    ], dim=0).to(device)
    env_states = torch.stack([
        torch.tensor(envs.envs[i].unwrapped.gen_gt_obs(), dtype=torch.float32)
        for i in range(args.num_envs)
    ], dim=0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.track:
        # dump args to config file
        config_save_path = os.path.join(wandb.run.dir, "config.yaml")
        with open(config_save_path, "w") as f:
            arg_vars = vars(args)
            arg_vars["teacher_model_class"] = type(teacher_model).__name__
            yaml.dump(arg_vars, f)

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
        steps_since_reset.zero_()

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
            Take student step
            """
            # generate a path plan from current state
            # start_planning_time = time.time()
            student_action = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            for i, env in enumerate(envs.envs):
                _, planned_actions = get_astar_plan(
                    agent_map=env.unwrapped.get_agent_map(visible_things=student_visibles_list[i]), 
                    true_map=env.unwrapped.get_true_map(),
                    agent_pos=env.unwrapped.agent_pos,
                    agent_dir=env.unwrapped.agent_dir,
                    goal_pos=env.unwrapped.goal_pos,
                    env_action_enum=env.unwrapped.actions,
                    agent_actions=create_custom_action_enum(STUDENT_ACTIONS[student_ids[i]]),
                    overlappable_ids=env.unwrapped.get_overlappable_ids(),
                )
                student_action[i] = planned_actions[0]
                # print("planning time ", time.time() - start_planning_time)
                # start_planning_time = time.time()
            
            # take environment step
            _, student_reward, terminations, truncations, student_infos = envs.step(student_action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)
            dones_buffer[:,step] = next_done.detach()
            student_reward = torch.tensor(student_reward, dtype=torch.float32, device=device)
            student_rewards_buffer[:,step] = student_reward.detach()

            steps_since_reset += 1

            """
            Get input to VQVAE
            """
            # encode student trajectory into zq
            # concatenate teacher state, student action, and student reward into a single step of student trajectory, add to buffer
            traj_step = torch.cat([
                env_states, 
                student_action.float().unsqueeze(1), 
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
            if args.z_flatten_method == "last_step":
                zq_collapsed = z_q[:, traj_chunk.shape[1]-1] # TODO - currently last step. change to running mean
            elif args.z_flatten_method == "mean":
                zq_collapsed = (mask.unsqueeze(-1) * z_q).mean(dim=1)
            zq_buffer[:,step] = zq_collapsed
            
            """
            Take teacher action
            """
            # get true env state after the student step
            env_states = torch.stack([
                torch.tensor(envs.envs[i].unwrapped.gen_gt_obs(), dtype=torch.float32)
                for i in range(args.num_envs)
            ], dim=0).to(device)
            env_state_buffer[:,step] = env_states

            with torch.no_grad():
                teacher_output = teacher_model(zq_collapsed, prev_teacher_action, env_states)
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
            # print("\n\n\n")
            student_values = {sid: np.zeros(args.num_envs) for sid in all_student_ids}
            dist_to_goal = {sid: np.zeros(args.num_envs) for sid in all_student_ids}
            for i, env in enumerate(envs.envs):
                for sid in all_student_ids:
                    # generate plan for each student from the current state
                    full_trace, _ = get_astar_plan(
                        agent_map=env.unwrapped.get_agent_map(visible_things=student_visibles_list[i]), 
                        true_map=env.unwrapped.get_true_map(),
                        agent_pos=env.unwrapped.agent_pos,
                        agent_dir=env.unwrapped.agent_dir,
                        goal_pos=env.unwrapped.goal_pos,
                        env_action_enum=env.unwrapped.actions,
                        agent_actions=create_custom_action_enum(STUDENT_ACTIONS[sid]),
                        overlappable_ids=env.unwrapped.get_overlappable_ids(),
                    )
                    # print("\nstudent_actions", STUDENT_ACTIONS[sid])
                    # print(f"env {i}, student {sid}, start {env.unwrapped.agent_pos}, goal {env.unwrapped.goal_pos}")
                    # print(full_trace)
                    # get distance to goal after n_steps_for_voi steps
                    if len(full_trace) > args.n_steps_for_voi + 1:
                        n_steps_taken = args.n_steps_for_voi
                        new_agent_pos = full_trace[n_steps_taken+1]['pos']

                    else:
                        # agent reached goal before n_steps_for_voi_steps
                        n_steps_taken = len(full_trace)
                        new_agent_pos = env.unwrapped.goal_pos

                    n_grids_to_goal = env.unwrapped.compute_n_grids_to_goal(start_pos=new_agent_pos)  
                    dist_to_goal[sid][i] = n_grids_to_goal
                    # print("new pos", new_agent_pos)
                    # print("taken", n_steps_taken)
                    # print("dist to goal", n_grids_to_goal)
                    # compute VoI
                    student_values[sid][i] = -n_grids_to_goal + args.n_steps_for_voi - n_steps_taken

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

            # sample new students for terminated environments
            for i in range(args.num_envs):
                if next_done[i]:
                    envs.envs[i].reset()
                    student_ids[i] = np.random.choice(all_student_ids)
                    student_visibles_list[i] = STUDENT_VISIBLE_THINGS[student_ids[i]]
                    envs.envs[i].unwrapped.set_obs_type(student_visibles_list[i])
                    env_states[i] = torch.tensor(envs.envs[i].unwrapped.gen_gt_obs(), dtype=torch.float32).to(device)
                    steps_since_reset[i] = 0

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
            # breakpoint()

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