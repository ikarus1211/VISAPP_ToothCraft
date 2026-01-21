import mesh2sdf
import numpy as np
from noise import snoise3

def add_noise_to_sdf(sdf_volume, world_bounds=(-1.0, 1.0), freq=2.8, amp=0.06, octaves=1):
    """
    Adds 3D Simplex noise to an SDF volume.

    Parameters:
        sdf_volume (np.ndarray): The input SDF of shape (N, N, N)
        world_bounds (tuple): Min and max world coordinate values
        freq (float): Frequency of noise
        amp (float): Amplitude of noise to add
        octaves (int): Number of octaves for fractal noise

    Returns:
        np.ndarray: Noisy SDF volume (same shape)
    """
    N = sdf_volume.shape[0]
    lin = np.linspace(world_bounds[0], world_bounds[1], N)
    x, y, z = np.meshgrid(lin, lin, lin, indexing='ij')

    noise_volume = np.zeros_like(sdf_volume)

    # Vectorized noise generation using numpy loops
    for i in range(N):
        for j in range(N):
            for k in range(N):
                noise_volume[i, j, k] = snoise3(
                    x[i, j, k] * freq,
                    y[i, j, k] * freq,
                    z[i, j, k] * freq,
                    octaves=octaves
                )


    # Scale and add noise to the SDF
    noisy_sdf = sdf_volume + amp * noise_volume
    return noisy_sdf, amp * noise_volume


def mesh_to_sdf(mesh, resolution):
    sdf =  mesh2sdf.compute(mesh.vertices, mesh.faces, size=resolution)
    return sdf


def normalize_trimesh(new_mesh, eta=0.95):
    centroid = new_mesh.centroid
    new_mesh.apply_translation(-centroid)

    distances = np.linalg.norm(new_mesh.vertices, axis=1)  # Distances from the origin
    max_distance = np.max(distances)

    if max_distance > 0:  # Avoid division by zero
        scale_factor = (1.0 / max_distance) * eta
        new_mesh.apply_scale(scale_factor)
    else:
        raise ValueError(f"Division by zero: max_distance = {max_distance}")

    return new_mesh, (centroid, scale_factor)

def map_voxel_grid_with_translation(grid_a, grid_b, translation, bounds):
    """
    Map values from voxel grid A to voxel grid B using a translation vector.

    Parameters:
    - grid_a: numpy array representing the source voxel grid (3D)
    - grid_b: numpy array representing the destination voxel grid (3D)
    - translation: tuple or array (tx, ty, tz) representing the translation vector


    Returns:
    - grid_b: Updated grid_b with mapped values from grid_a
    """
    # Grid dimensions
    dim_x, dim_y, dim_z = grid_a.shape

    voxel_size = (np.asarray(bounds[1]) - np.asarray(bounds[0])) / dim_x


    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                # Calculate translated coordinates
                new_x = int(round(x + translation[0] / voxel_size[0]))
                new_y = int(round(y + translation[1] / voxel_size[1]))
                new_z = int(round(z + translation[2] / voxel_size[2]))
                # Check if new coordinates are within bounds of grid B
                if 0 <= new_x < grid_b.shape[0] and 0 <= new_y < grid_b.shape[1] and 0 <= new_z < grid_b.shape[2]:
                    # Map the value from grid A to the new position in grid B
                    grid_b[new_x, new_y, new_z] = grid_a[x, y, z]

    return grid_b



def sdf_union(sdf1, sdf2):
    return np.minimum(sdf1, sdf2)

def sdf_difference(sdf1, sdf2):
    return np.maximum(sdf1, -sdf2)

def sdf_intersection(sdf1, sdf2):
    return np.maximum(sdf1, sdf2)


# def tsdf_diff(tsdf_A, tsdf_B):
#
#     dim_x, dim_y, dim_z = tsdf_A.shape
#
#     res_grid = np.zeros(tsdf_A.shape)
#     for x in range(dim_x):
#         for y in range(dim_y):
#             for z in range(dim_z):
#                 valA = tsdf_A[x, y, z]
#                 valB = tsdf_B [x, y, z]
#                 res_grid[x,y,z] = max(valA, -valB)
#
#     return res_grid