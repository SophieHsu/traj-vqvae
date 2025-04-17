import os
import argparse
import h5py


AGENTS = { 
    0: ["all", "lrf"],
    1: ["all", "lf"],
    2: ["all", "rf"],
    3: ["nored", "lrf"],
    4: ["nored", "lf"],
    5: ["nored", "rf"],
}

def get_agent_id(filename):
    for agent_id, substrings in AGENTS.items():
        if all(sub in filename for sub in substrings):
            return agent_id
    return None 

def combine_datasets(data_dir, save_path):
    # check if save_path already exists
    if os.path.exists(save_path):
        print(f"filepath {save_path} exists. continue to overwrite")
        os.remove(save_path)
    
    # create new file
    combined_f = h5py.File(save_path, "a")
    combined_data = combined_f.create_group("trajectories")
    n_traj = 0

    for file in os.listdir(data_dir):
        # Read a file and get data
        filepath = os.path.join(data_dir, file)
        
        # get agent ID
        agent_id = get_agent_id(file)
        if agent_id is None:
            print(f"no matching agent for file {file}. skipping")
            continue
        
        with h5py.File(filepath, 'r') as f:
            print(f"working on {filepath}")
            data = f["trajectories"]

            # copy data to new dataset, along with agent IDs
            for traj_idx in data.keys():
                traj_data = data[traj_idx]
                new_traj_group = combined_data.create_group(f"traj_{n_traj}")

                # Copy all datasets
                for dataset_name in traj_data.keys():
                    traj_data.copy(dataset_name, new_traj_group)

                # Copy attributes
                for attr_name, attr_value in traj_data.attrs.items():
                    new_traj_group.attrs[attr_name] = attr_value
                new_traj_group.attrs["agent_id"] = agent_id  # Add agent ID

                n_traj += 1

def main(args):
    combine_datasets(data_dir=args.data_root_dir, save_path=args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root-dir", type=str, help="directory containing hdf5 files to combine")
    parser.add_argument("--save-path", type=str, help="path to save combined hdf5 file to")
    args = parser.parse_args()
    main(args)