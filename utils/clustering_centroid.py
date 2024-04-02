"""
Created on 18 April 2023 1:23 AM
@author: nabin

"""
import math
import ast
import os


prob_dic_aa = dict()
prob_dic_sec = dict()
prob_dic_atom = dict()


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def distance(p1, p2):
    dis =  math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    return dis


def create_clusters(points, thres):
    clusters = []
    while points:
        # Select a point randomly and assign it to a new cluster
        cluster = [points.pop(0)]
        # Iterate through the rest of the points and add them to the cluster if they are within the threshold distance
        i = 0
        while i < len(points):
            if distance(cluster[0], points[i]) <= thres:
                cluster.append(points.pop(i))
            else:
                i += 1
        clusters.append(cluster)
    return clusters


def centroid(file, save_cords, save_probs_aa , thres, save_ca_probs):
    # Read the data from the file
    points = []
    with open(file, 'r') as f:
        # Read all the lines in the file
        lines = f.readlines()

    for line in lines:
        vals = line.split(" ")
        for limiter in vals:
            if limiter == '':
                vals.remove(limiter)
        vals = list(filter(lambda x: x != '', vals))               
        points.append(Point(float(vals[0]), float(vals[1]), float(vals[2])))

    # Create the clusters
    clusters = create_clusters(points, thres=thres)

    with open(save_probs_aa,'w') as p:
        with open(save_cords, 'w') as f:
            with open(save_ca_probs, 'w') as a_p:
                for i, cluster in enumerate(clusters):
                    x_sum = 0
                    y_sum = 0
                    z_sum = 0
                    num_points = len(cluster)
                    collect_values = list()
                    collect_values_sec = list()
                    collect_values_atom = list()
                    for point in cluster:
                        x_sum += point.x
                        y_sum += point.y
                        z_sum += point.z
                        cords = (point.x, point.y, point.z)
                        if cords in prob_dic_aa:
                            values = prob_dic_aa.get(cords)
                            collect_values.append(values) 
                            atom_values = prob_dic_atom.get(cords)
                            collect_values_atom.append(atom_values)
                        if cords in prob_dic_sec:
                            values = prob_dic_sec.get(cords)
                            collect_values_sec.append(values)

                    averages = list()
                    averages_atom = list()
                    for i in range(len(collect_values[0])):
                        total = sum(collect_values[j][i] for j in range(len(collect_values)))
                        average = total / len(collect_values)     
                        averages.append(average) 
                    averages = ' '.join(str(x) for x in averages)
                    for i in range(len(collect_values_atom[0])):
                        total = sum(collect_values_atom[j][i] for j in range(len(collect_values)))
                        average_atm = total / len(collect_values_atom)     
                        averages_atom.append(average_atm) 
                    averages_atom = ' '.join(str(x) for x in averages_atom)
                    print(averages, file=p)
                    print(averages_atom, file=a_p)   
                    x_avg = x_sum / num_points
                    y_avg = y_sum / num_points
                    z_avg = z_sum / num_points
                    print(f'{x_avg} {y_avg} {z_avg}', file=f)
                    

def proc_probabilities_aa(file, file_atom):
    with open(file, 'r') as f:
        # Read all the lines in the file
        line = f.readline()
        while line:
            line_c = ast.literal_eval(line)
            key = tuple(line_c[0])
            vals = line_c[1:]
            prob_dic_aa[key] = vals
            line = f.readline()
    with open(file_atom, 'r') as f:
        line = f.readline()
        while line:
            line_c = ast.literal_eval(line)
            key = tuple(line_c[0])
            vals = line_c[1:]
            prob_dic_atom[key] = vals
            line = f.readline()

    

def main(config_dict):
    cord_data = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_coordinates_ca.txt"
    cord_probs_aa = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_emi.txt"
    cords_prob_atom = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_ca_prob.txt" 
    save_cords = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca.txt"
    save_probs_aa = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_emission_aa_ca.txt"
    save_ca_probs = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca_probs.txt"
    if os.path.exists(save_cords):
        os.remove(save_cords)

    if os.path.exists(save_probs_aa):
        os.remove(save_probs_aa)

    proc_probabilities_aa(cord_probs_aa, cords_prob_atom)
    centroid(cord_data, save_cords, save_probs_aa, config_dict['clustering_threshold'], save_ca_probs)
    return save_cords, save_probs_aa, save_ca_probs