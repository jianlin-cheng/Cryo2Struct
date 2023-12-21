"""
Created on 04 Sep 2023 06:16 AM
@author: nabin

"""


import argparse
import yaml
import os
import shutil
import threading

from utils import get_probs_cords_from_atom_amino, clustering_centroid, grid_division
from viterbi import alignment
import subprocess

import warnings
warnings.filterwarnings("ignore")


script_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{script_dir}/config/arguments.yml"
COMMENT_MARKER = '#'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        default=config_file_path)
    parser.add_argument('--density_map_name', type=str)
    
    return parser.parse_args()


def process_arguments(args):
    if args.config is not None:
        config_dict = yaml.safe_load(args.config)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith(COMMENT_MARKER)}
        args.config = args.config.name
    else:
        config_dict = dict()
    
    if args.density_map_name is not None:
        config_dict['density_map_name'] = args.density_map_name
    return config_dict


def delete_directory(directory_path):
    shutil.rmtree(directory_path)
           
    
def make_predictions(config_dict):
    grid_division.create_subgrids(input_data_dir=config_dict['input_data_dir'], density_map_name=config_dict['density_map_name'])
    print("\nCryo2Struct DL: Grid Division Complete!")
    density_map_dir = os.path.join(config_dict['input_data_dir'],config_dict['density_map_name'])
    density_map_split_dir = os.path.join(density_map_dir, f"{config_dict['density_map_name']}_splits")
    script_name = ['../../infer/atom_inference.py', '../../infer/amino_inference.py']
    checkpoint_name = ['atom_checkpoint', 'amino_checkpoint']

    for s in range(len(script_name)):
        cmd = ['python3', script_name[s], density_map_split_dir, str(config_dict['input_data_dir']),
               str(config_dict['density_map_name']),  str(config_dict[checkpoint_name[s]]) , config_dict['infer_run_on'], str(config_dict['infer_on_gpu'] )]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            if return_code == 0:
                print(f"Cryo2Struct DL: Prediction {s + 1} / {len(script_name)} Complete!")
                # print(stdout)
            else:
                print(f"Cryo2Struct Deep Learning Block failed with exit code {return_code}.")
                print("Standard Error:")
                print(stderr)
        except subprocess.CalledProcessError as e:
            print(f"Cryo2Struct Deep Learning Block failed with exit code {e.returncode}.")
            print("Standard Error:")
            print(e.stderr)
        except Exception as e:
            print(f"An error occurred in Cryo2Struct Deep Learning Block: {str(e)}")

    delete_thread = threading.Thread(target=delete_directory, args=(density_map_split_dir,))
    delete_thread.start() # runs in background to delete the grid division directory
    delete_thread1 = threading.Thread(target=delete_directory, args=(f"{density_map_dir}/lightning_logs",))
    delete_thread1.start()

def extract_probs_cords_from_atom_amino(config_dict):
    probability_file_atom = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_atom.txt" # comes from atom_inference.py
    probability_file_amino = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino.txt" # comes from amino_inference.py
    probability_file_amino_atom_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_emi.txt" # save common amino and atom
    probability_file_amino_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_emi.txt" # save amino probability as emission
    
    save_cords = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_coordinates_ca.txt" # save cords as transition matrix

    if os.path.exists(save_cords):
        os.remove(save_cords)
    
    if os.path.exists(probability_file_amino_atom_common_emi):
        os.remove(probability_file_amino_atom_common_emi)

    if os.path.exists(probability_file_amino_common_emi):
        os.remove(probability_file_amino_common_emi)
    
    get_probs_cords_from_atom_amino.get_joint_probabity_common_threshold(probability_file_atom=probability_file_atom, probability_file_amino_atom_common=probability_file_amino_atom_common_emi, 
                    probability_file_amino=probability_file_amino, s_c=save_cords, threshold = config_dict['threshold'])


def cluster_emission_transition(config_dict):
    save_cords, save_probs_aa= clustering_centroid.main(config_dict)
    return save_cords, save_probs_aa


def main():
    args = parse_arguments()
    config_dict = process_arguments(args)
    print("\n##############- Cryo2Struct -##############")
    print("\nRunning with below configuration: ")
    for key,value in config_dict.items():
        print("%s : %s"%(key, value))
    print("\n- This might take a bit. Time for a coffee break, maybe! -")
    make_predictions(config_dict)
    # preparing for HMM model 
    extract_probs_cords_from_atom_amino(config_dict)
    # clustering and preparing emission and transition matrix
    coordinate_file, emission_file = cluster_emission_transition(config_dict)
    # run viterbi algorithm
    alignment.main(coordinate_file, emission_file, config_dict)

if __name__ == "__main__":
    main()
