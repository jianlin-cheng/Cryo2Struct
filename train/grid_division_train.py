import numpy as np
import mrcfile
import os
import math
from copy import deepcopy
import sys

box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundry issues


def create_manifest(full_image):
    # creates a list of box_size tensors. Each tensor is passed to Transformer Unet independently
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size))
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = full_image
    manifest = list()

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        manifest.append(next_chunk)
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point  # Reset
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset
    return manifest


def get_data(density_map_dir):
    protein_manifest = None
    amino_manifest = None
    atom_manifest = None
    processed_maps = [m for m in os.listdir(density_map_dir)]
    for maps in range(len(processed_maps)):
        os.chdir(density_map_dir)
        if processed_maps[maps] == "emd_normalized_map.mrc":
            p_map = mrcfile.open(processed_maps[maps], mode='r')
            protein_data = deepcopy(p_map.data)
            protein_manifest = create_manifest(protein_data)
        
        if processed_maps[maps] == "atom_emd_normalized_map.mrc":
            atom_map = mrcfile.open(processed_maps[maps], mode='r')
            atom_data = deepcopy(atom_map.data)
            atom_manifest = create_manifest(atom_data) 

        if processed_maps[maps] == "amino_emd_normalized_map.mrc":
            amino_map = mrcfile.open(processed_maps[maps], mode='r')
            amino_data = deepcopy(amino_map.data)
            amino_manifest = create_manifest(amino_data)

    return protein_manifest, atom_manifest, amino_manifest


def reconstruct_map(manifest, image_shape):
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimensions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2]))
    counter = 0
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]
                counter += 1
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image


def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions


def create_subgrids(input_data_dir, grid_division_dir):

    density_map_names = [m for m in os.listdir(input_data_dir)]
    for density_map_name in density_map_names:
        density_map_dir = os.path.join(input_data_dir,density_map_name)

        protein, atom, amino = get_data(density_map_dir)
        if protein is not None and atom is not None and amino is not None:
            for i in range(len(protein)):
                save_file_name = f'{grid_division_dir}/{density_map_name}_{i}.npz'
                np.savez_compressed(file=save_file_name, protein_grid=protein[i], atom_grid=atom[i], amino_grid=amino[i])
        else:
            print("There is no input map. Please check the input density map's directory")
            exit()


if __name__ == "__main__":
    input_data_dir = sys.argv[1]
    grid_division_dir = sys.argv[2]
    os.makedirs(grid_division_dir, exist_ok=True)
    create_subgrids(input_data_dir=input_data_dir, grid_division_dir=grid_division_dir)
    
