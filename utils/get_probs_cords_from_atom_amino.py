"""
Created on 02 March 2023 05:14:00 PM
@author: nabin

"""
import ast
import math

ca_coordinates = list()
prob_dic = dict()


def get_joint_probabity_common_threshold(probability_file_atom, probability_file_amino_atom_common, probability_file_amino, s_c, threshold, probability_file_amino_atom_common_ca_prob):
    """
    get only common carbon alphas from amino and atom files
    
    """
    common_ca = dict()
    common_coordinate_prob = dict()
    amino_acid_emission = dict()
    count_uncommon_atoms = 0
    count_common_atoms = 0
    total_atom_entries = 0
    total_saved_ca = 0

    with open(probability_file_amino, 'r') as amino_prob:
        for line in amino_prob:
            line_a = ast.literal_eval(line)
            common_coordinate_prob[tuple(line_a[0])] = 1 - line_a[1]
            aa_val = list(line_a[2:])
            equal_part_add = line_a[1]  / 20
            aa_val = tuple([x + equal_part_add for x in aa_val])
            amino_acid_emission[tuple(line_a[0])] = aa_val
            total_atom_entries += 1
                
            
    with open(probability_file_atom, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line)
            try:
                common_ca[tuple(line_a[0])] = math.sqrt(common_coordinate_prob[tuple(line_a[0])] * line_a[2])
                common_ca[tuple(line_a[0])] = line_a[2]
                count_common_atoms += 1 
    
            except KeyError:
                count_uncommon_atoms += 1

    save_cluster_co = open(s_c, 'a')
    save_cluster_prob = open(probability_file_amino_atom_common_ca_prob,'a')
    amino_atom_prob = open(probability_file_amino_atom_common,'a')
    for k,v in common_ca.items():
        if v > threshold:
            try:
                emiss_val = amino_acid_emission[k]
                x,y,z = k
                print(f"{x} {y} {z}", file=save_cluster_co)
                amino_atom_prob.write(f"{list(k)}")
                save_cluster_prob.write(f"{list(k)}")
                save_cluster_prob.write(f", {v}")
                save_cluster_prob.write(f"\n")
                for e in emiss_val:
                    amino_atom_prob.write(f", {e}")
                amino_atom_prob.write(f"\n")
                total_saved_ca += 1
            except KeyError:
                q  = 1

