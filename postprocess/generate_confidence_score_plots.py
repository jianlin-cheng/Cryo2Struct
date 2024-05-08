"""
Created on 8 May 2024 12:23 PM
@author: nabin

Usage:
- generate pdb with color spectrum
"""
from Bio import PDB
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

def save_scores_to_pdb(conf_score_file, input_pdb_file, output_pdb_file):
    conf_df = pd.read_csv(conf_score_file)
    scores = conf_df['Pred AA Prob'].to_list()

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb_file)

    # Iterate through atoms and assign scores as B-factors
    for atom in structure.get_atoms():
        residue_id = atom.get_parent().get_id()[1]
        score = scores[residue_id]
        atom.set_bfactor(score)

    # Write modified structure to output PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file)
    if os.path.exists(input_pdb_file):
        os.remove(input_pdb_file)

    
def generate_plot(conf_score_file, plot_filename):
    shapes = ['o', 's', '^', 'x', 'v', 'D', 'p', '*', '>', '<', 'h', '+', '|', '.', '1', '2', '3', '4', '8', 'd']
    df = pd.read_csv(conf_score_file)
    residue = df['Residue'].to_list()
    ca_confidence = df['Pred CA Prob'].to_list()
    aa_confidence = df['Pred AA Prob'].to_list()
    unique_residues = list(set(residue))
    num_unique_residues = len(unique_residues)

    correlation_coeff, p_value = pearsonr(ca_confidence, aa_confidence)
    avg_ca_conf = sum(ca_confidence)/len(ca_confidence)
    avg_aa_conf = sum(aa_confidence)/len(aa_confidence)
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, res in enumerate(unique_residues):
        shape = shapes[i % num_unique_residues]  # Cycle through shapes
        indices = [idx for idx, r in enumerate(residue) if r == res]
        plt.scatter([ca_confidence[idx] for idx in indices], [aa_confidence[idx] for idx in indices], label=res, marker=shape, edgecolors='black', s=100)

    plt.title('Residue Scores')
    plt.xlabel('CA Confidence')
    plt.ylabel('Amino Acid Type Confidence')
    plt.legend(title='Residue')
    plt.grid(False)  # Turn off grid
    # plt.xlim(0.4, 0.7)  # Set x-axis limit
    # plt.ylim(0, 1)  # Set y-axis limit
    plt.tight_layout()
    # plt.show()

    plt.savefig(plot_filename, bbox_inches='tight', dpi=1000)
    return avg_ca_conf, avg_aa_conf