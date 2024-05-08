""" 
Created on 25 Jan 2023 10:21 AM
Updated on 8 May 2024 3:43 PM
@author: nabin

Usage:
- Construct HMM
- Align using Viterbi

"""

import numpy as np
import scipy.stats
import os
import ctypes
import time
import glob
from utils import extract_seq_from_pdb
from postprocess import generate_confidence_scores, generate_confidence_score_plots


one_to_three_amino ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN',
'D':'ASP', 'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 'Y':'TYR',  
'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',
'G':'GLY', 'P':'PRO', 'C':'CYS'}


residue_label = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
}

aa_probs_train = [0.07943948021002176, 0.0497783411606611, 0.04433904104844631, 0.05350186331961819, 0.022626103099067648, 0.03937526841500076, 0.05627820955072524, 0.07217724394940638, 0.01882610864053863, 0.06108044830500256, 0.09799918263302994, 0.05331463086876411, 0.021703725253868638, 0.045350220966155465, 0.041743693113336935, 0.06416670130086032, 0.058655985481346026, 0.012496017067730628, 0.03598361109956638, 0.071164124516853]
exclude_states = list()
chain_id_states = dict()
hmm_probability = list()
seq_key_list = list()
chains_sequence_dict = dict()
seq_list = list()
chains_sec_sequence_dict = dict()
transition_dic = dict()
hmm_dic = dict()
chain_list = list()
cord_idx_prob_dict = dict()

chain_count = 0

start_time = time.time()



def load_data(trans_file, hmm_file, save_ca_probs):
    trans_count = 0
    hmm_count = 0
    ca_prob_count = 0
    # creating key-value pair for hmm_file
    with open(hmm_file, 'r') as h_file:
        for line in h_file:
            hmm_count += 1
            h = line.strip()
            h = h.split()
            h_value = int(h[1].replace("\n", ""))
            hmm_dic[f'{h[0]}_{hmm_count}'] = h_value
        
    # creating key-value pair for trans_file
    with open(trans_file, 'r') as t_file:
        for line in t_file:
            t = line.replace("\n", "")
            transition_dic[trans_count] = t
            trans_count += 1

    with open(save_ca_probs, 'r') as ca_prob_f:
        for line in ca_prob_f:
            p = line.replace("\n", "")
            cord_idx_prob_dict[ca_prob_count] = p
            ca_prob_count += 1


def save(save_filename):
    count = 0
    with open(save_filename, 'a') as fi:
        fi.write("Author Cryo2Struct\n")
    for key, value in hmm_dic.items():
        atom = "CA"
        residue_name =  one_to_three_amino[key.split("_")[0]]
        cord_idx = value # gaps are not included
        if cord_idx >= 0 and cord_idx < len(transition_dic):
            xyz = transition_dic[cord_idx].split(" ")
            while '' in xyz:
                xyz.remove("")
            
            x = round(float(xyz[0]), 3)
            y = round(float(xyz[1]), 3)
            z = round(float(xyz[2]), 3)
            with open(save_filename, 'a') as fi:
                fi.write('ATOM')
                fi.write('  ')
                fi.write(str(count).rjust(5))
                fi.write('  ')
                fi.write(atom.ljust(4))
                fi.write(residue_name.rjust(3))
                fi.write(' ')
                fi.write(f'{chain_list[count]}')
                fi.write(str(count).rjust(4))
                fi.write('    ')
                fi.write(str(x).rjust(8))
                fi.write(str(y).rjust(8))
                fi.write(str(z).rjust(8))
                fi.write(str(1.00).rjust(5))
                fi.write(str(0.00).rjust(5))
                fi.write('           ')
                fi.write(atom[0:1].rjust(1))
                fi.write('  ')
                fi.write('\n')
            count += 1
        
        
def makeEmission(emission_file, length_coordinate_list):
    emi_matrix = np.zeros((length_coordinate_list, 20), dtype=np.double)
    with open(emission_file,"r") as emission_f:
        idx = 0
        for line in emission_f:
            vals = line.split()
            for l in range(len(vals)):
                emi_matrix[idx][l] = vals[l]
            idx += 1
    return emi_matrix
                
    
def makeEmission_aa(emission_mat):
    for em in range(len(emission_mat)):
        for aa_em in range(len(emission_mat[em])):
            emission_mat[em][aa_em] = np.sqrt(np.double(emission_mat[em][aa_em] * aa_probs_train[aa_em])) # geometric mean
    return emission_mat


def normalize_sum(coordinate_distance_matrix):
    coordinate_distance_matrix = coordinate_distance_matrix / coordinate_distance_matrix.sum(axis=1, keepdims=True)
    return coordinate_distance_matrix


def probability_density_function(coordinate_distance_matrix_dis, std_lambda):
    computed_mean = 3.8047179727719045
    computed_std = 0.03622304 * std_lambda
    p_norm = scipy.stats.norm(computed_mean,computed_std)
    probability_density_matrix  = p_norm.pdf(coordinate_distance_matrix_dis) # type: ignore
    return probability_density_matrix
    
def make_standard_observations(chain_obser):
    observations = tuple(chain_obser.strip('\n'))
    non_standard_amino_acids = ['X', 'U', 'O']
    filtered_observations = list()
    for o in observations:
        if o in non_standard_amino_acids:
            # print(f" - Removed {o}")
            pass
        else:
            filtered_observations.append(o)
    filtered_observations = list(tuple(filtered_observations))
    seq_list.extend(filtered_observations)
    return filtered_observations

def run_vitebi(key_idx, chain_observations, transition_matrix, emission_matrix, states, initial_matrix, config_dict, save_ca_probs, emission_matrix_dl):
    print(f"Cryo2Struct Alignment: Aligning Chain {seq_key_list[key_idx]}")
    chain_observations_np = np.array([residue_label[x] for x in chain_observations], dtype=np.int32)
    exclude_states_np = np.array(exclude_states, dtype=np.int32)

    transition_matrix_log = np.log(transition_matrix)
    emission_matrix_log = np.log(emission_matrix)
    initial_matrix_log = np.log(initial_matrix)


    states_len = len(states)
    exclude_arr_len = len(exclude_states_np)
    chain_arr_len = len(chain_observations_np)

    transition_arr = (ctypes.c_double * (states_len * states_len))()
    emission_arr = (ctypes.c_double * (states_len * 20))()
    initial_arr = (ctypes.c_double * len(initial_matrix_log))()
    exclude_arr = (ctypes.c_int * len(exclude_states_np))()
    chain_arr = (ctypes.c_int * len(chain_observations_np))()


    for i in range(states_len):
        for j in range(states_len):
            transition_arr[i*states_len + j] = transition_matrix_log[i,j]
    
    for i in range(states_len):
        for j in range(20):
            emission_arr[i*20 + j] = emission_matrix_log[i,j]
    

    for i in range(len(initial_matrix_log)):
        initial_arr[i] = initial_matrix_log[i]


    for i in range(len(exclude_states_np)):
        exclude_arr[i] = exclude_states_np[i]

    for i in range(len(chain_observations_np)):
        chain_arr[i] = chain_observations_np[i]

    

    # Load the C++ shared library
    viterbi_algo_path = os.path.abspath(config_dict['input_data_dir'])
    viterbi_algo_path = os.path.dirname(viterbi_algo_path)
    lib = ctypes.cdll.LoadLibrary(f'{viterbi_algo_path}/viterbi/viterbi.so')

    # Define the C++ wrapper function
    wrapper_function = lib.viterbi_main

    # Define the argument types for the wrapper function
    wrapper_function.argtypes = [
    ctypes.POINTER(ctypes.c_int), # chain_o
    ctypes.c_int, # chain_o_len
    ctypes.c_int, # num_states
    ctypes.POINTER(ctypes.c_double), # transition_matrix_log
    ctypes.POINTER(ctypes.c_double), # emission_matrix_log
    ctypes.POINTER(ctypes.c_double), # initial_matrix_log
    ctypes.POINTER(ctypes.c_int), # exclude_arr
    ctypes.c_int, # exclude_arr_len
    ]

    wrapper_function.restype = ctypes.POINTER(ctypes.c_int)
    results = wrapper_function(chain_arr, chain_arr_len, states_len, transition_arr, emission_arr, initial_arr, exclude_arr, exclude_arr_len)
    observation_length_for_c = len(chain_observations)
    exclude_state_from_c = np.ctypeslib.as_array(results, shape=(observation_length_for_c,))
    exclude_states.extend(exclude_state_from_c)

    key_idx += 1
    if key_idx < len(seq_key_list):
        execute(key_idx=key_idx, states=states,transition_matrix=transition_matrix, emission_matrix=emission_matrix, config_dict=config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)
    else:
        cord_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca.txt"
        hmm_out_save_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_hmm_{config_dict['use_sequence']}.txt"
        save_pdb_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_{config_dict['use_sequence']}.pdb"
        conf_score_pdb_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_{config_dict['use_sequence']}_conf_score.pdb"
        save_confidence_score = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_confidence_scores.csv"
        save_prob_score = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_prob_scores.csv"
        save_conf_score_plot = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_conf_scores.png"
        if os.path.exists(save_confidence_score):
            os.remove(save_confidence_score)

        if os.path.exists(save_prob_score):
            os.remove(save_prob_score)

        if os.path.exists(hmm_out_save_file):
            os.remove(hmm_out_save_file)

        if os.path.exists(save_pdb_file):
            os.remove(save_pdb_file)

        if os.path.exists(conf_score_pdb_file):
            os.remove(conf_score_pdb_file)

        if os.path.exists(save_conf_score_plot):
            os.remove(save_conf_score_plot)
            
        hmm_outs = open(hmm_out_save_file, 'a')
        for i in range(len(exclude_states)):     
            print(f"{seq_list[i]}\t{exclude_states[i]}",file=hmm_outs)
        hmm_outs.close()
        load_data(trans_file=cord_file, hmm_file=hmm_out_save_file, save_ca_probs=save_ca_probs)
        save(save_filename=save_pdb_file)
        print("Cryo2Struct Alignment: Total modeled residues:", len(set(exclude_states)))
        end_time = time.time()
        runtime_seconds = end_time - start_time
        runtime_minutes = runtime_seconds / 60
        print(f"Cryo2Struct Alignment: Run time {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

        # clean up:
        map_directory_path = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}"
        if os.path.exists(f"{map_directory_path}/{config_dict['density_map_name']}_amino_predicted.mrc"):
            os.remove(f"{map_directory_path}/{config_dict['density_map_name']}_amino_predicted.mrc")
        if os.path.exists(f"{map_directory_path}/{config_dict['density_map_name']}_atom_predicted.mrc"):
            os.remove(f"{map_directory_path}/{config_dict['density_map_name']}_atom_predicted.mrc")
        files_to_delete = glob.glob(os.path.join(map_directory_path, f"*.txt"))
        for f in files_to_delete:
            os.remove(f)
        print("Cryo2Struct: Finished!\n")
        ami_list = list()
        ca_list = list()
        seq_list_conf = list()

        for k,v in hmm_dic.items():
            amino = k.split("_")[0]
            seq_list_conf.append(amino)
            ami=  residue_label[amino]
            ami_list.append(emission_matrix[v][ami])
            ca_list.append(float(cord_idx_prob_dict[v]))

        generate_confidence_scores.res_prob_score_files(save_prob_score_file=save_prob_score, seq_list=seq_list, 
                                                        seq_list_conf=seq_list_conf, ca_list=ca_list, ami_list=ami_list)
        trained_regression_model_aa = f"{config_dict['confidence_score_models']}/aa_regression_model.pkl"
        trained_regression_model_ca = f"{config_dict['confidence_score_models']}/ca_regression_model.pkl"
        generate_confidence_scores.gen_conf_scores(prob_scores=save_prob_score, save_path=save_confidence_score,
                                                   trained_regression_model_aa=trained_regression_model_aa, 
                                                   trained_regression_model_ca=trained_regression_model_ca)
        
         
        generate_confidence_score_plots.save_scores_to_pdb(conf_score_file=save_confidence_score, 
                                                       input_pdb_file=save_pdb_file, 
                                                       output_pdb_file=conf_score_pdb_file)
        
        avg_ca_conf, avg_aa_conf = generate_confidence_score_plots.generate_plot(conf_score_file=save_confidence_score, plot_filename=save_conf_score_plot)
        
        
        print(f"+ Cryo2Struct Outputs: ")
        print(f"Average carbon-alpha and amino acid-type confidence score are {avg_ca_conf} and {avg_aa_conf}, respectively.")
        print("Modeled Structure saved path:")
        print(f"- {conf_score_pdb_file}")
        print("Confidence Score csv file save path:") 
        print(f"- {save_confidence_score}")
        print("Confidence Score plot saved path:")
        print(f"- {save_conf_score_plot}")

        exit()

def execute(key_idx, states, transition_matrix, emission_matrix, config_dict, save_ca_probs, emission_matrix_dl):
    chain_sequence = chains_sequence_dict[seq_key_list[key_idx]]
    chain_observations = make_standard_observations(chain_obser=chain_sequence)
    initial_hidden_pobabilities = np.zeros((len(states)), dtype=np.double)
    observation_seq_first_amino_count = 0
    for i_c in range(len(states)):
        observation_seq_first_amino_count += emission_matrix[i_c][residue_label[chain_observations[0]]] 
    for i_c in range(len(states)):
        initial_hidden_pobabilities[i_c] = emission_matrix[i_c][residue_label[chain_observations[0]]] / observation_seq_first_amino_count
    run_vitebi(key_idx=key_idx, chain_observations=chain_observations ,transition_matrix=transition_matrix, emission_matrix=emission_matrix, 
               states=states, initial_matrix=initial_hidden_pobabilities, config_dict=config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)



def main(coordinate_file, emission_file, config_dict, save_ca_probs):
    fasta_file = [f for f in os.listdir(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}") if f.endswith(".fasta")]
    fasta_file.sort()
    if config_dict['use_sequence'] == "full":
        sequence_file = fasta_file[0]
        print("Cryo2Struct Alignment: Running with full fasta sequence")
    else:
        pdb_name = fasta_file[0].split(".")[0]
        pdb_name = pdb_name.split("_")[0]
        pdb_file_p = f"{pdb_name.lower()}.pdb"
        pdb_file_dir_p = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_p}"
        
        if os.path.exists(pdb_file_dir_p):
            pdb_file_dir = pdb_file_dir_p
        else:
            pdb_file_e = f"{pdb_name.lower()}"
            pdb_file_dir_ent = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_e}.ent"
            pdb_file_dir_pdb = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_e}.pdb"
            if os.path.exists(pdb_file_dir_ent):
                pdb_file_dir = pdb_file_dir_ent
            elif os.path.exists(pdb_file_dir_pdb):
                pdb_file_dir = pdb_file_dir_pdb

        # print(pdb_file_dir)
        atomic_seq_chain_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/atomic_seq_chain.fasta"
        atomic_seq_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/atomic_seq.fasta"
        extract_seq_from_pdb.extract_seq(pdb_file_dir, atomic_seq_chain_file, atomic_seq_file)
    
        if config_dict['use_sequence'] == "atomic_no_chain":
            sequence_file = atomic_seq_file 
            print("Running with ATOMIC NO CHAIN SEQUENCE")
        elif config_dict['use_sequence'] == "atomic_chain":
            sequence_file = atomic_seq_chain_file
            print("Running with CHAIN SEQUENCE")

    
    # read the coordinate file and append them to list
    coordinate_list = list()
    with open(coordinate_file,"r") as coordinate_f:
        for line in coordinate_f:
            x_y_z = [float(x) for x in line.split()]
            coordinate_list.append(x_y_z)
    
    # create a numpy array filled with zeros with size as coordinate file
    length_coordinate_list = len(coordinate_list)
    coordinate_distance_matrix = np.zeros((length_coordinate_list, length_coordinate_list), dtype=np.double)
    
    # compute distance between each carbon alpha to other and put into distance matrix
    for carbon_alpha in range(length_coordinate_list):
        for carbon_alpha_next in range(length_coordinate_list):
            coordinate_distance_matrix[carbon_alpha][carbon_alpha_next] = np.linalg.norm(np.array(coordinate_list[carbon_alpha]) - np.array(coordinate_list[carbon_alpha_next]))


    
    coordinate_distance_matrix_dis = coordinate_distance_matrix
    coordinate_distance_matrix = probability_density_function(coordinate_distance_matrix_dis, config_dict['std_lambda'])
    coordinate_distance_matrix += 1e-20
    coordinate_distance_matrix = normalize_sum(coordinate_distance_matrix)
    
    emission_matrix = makeEmission(emission_file, length_coordinate_list)
    emission_matrix += 1e-20
    emission_matrix = normalize_sum(emission_matrix)
    emission_matrix_dl = emission_matrix
    emission_matrix_aa = makeEmission_aa(emission_matrix)
    emission_matrix_aa = normalize_sum(emission_matrix_aa)

    assert coordinate_distance_matrix.shape == (length_coordinate_list, length_coordinate_list)
    for row in range(len(coordinate_distance_matrix[0])):
        assert abs(sum(coordinate_distance_matrix[row]) - 1) < 0.0001, f'Row {row} does not sum to 1 in transition matrix'
    
    assert emission_matrix_aa.shape == (length_coordinate_list, 20)
    for row in range(length_coordinate_list):
        assert abs(sum(emission_matrix_aa[row]) - 1) < 0.0001, f'Row {row} does not sum to 1 in emission matrix AMINO'

    states= list(tuple(idx for idx in range(length_coordinate_list)))

    with open(os.path.join(config_dict['input_data_dir'], config_dict['density_map_name'], sequence_file),"r") as seq_f:
        seq_lines = seq_f.readlines()
    
    for seq_contents in range(0,len(seq_lines),2):
        seq_c = seq_lines[seq_contents]
        seq_c = seq_c.split("|")[1]
        seq_c  = seq_c.split(" ")
        seq_c = seq_c[1:]
        for seq_chain in seq_c:
            seq_key = seq_chain.replace(",","").strip('\n')
            seq_key_list.append(seq_key)
            chains_sequence_dict[seq_key] = seq_lines[seq_contents + 1].strip('\n')
    for ke, va in chains_sequence_dict.items():
        length_va = len(va)
        chain_list.extend(ke*length_va)
    key_idx = 0
    print("Cryo2Struct Alignment: HMM Construction Complete!")
    execute(key_idx=key_idx, states=states,transition_matrix=coordinate_distance_matrix, emission_matrix=emission_matrix_aa, config_dict =config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)
    exit()
