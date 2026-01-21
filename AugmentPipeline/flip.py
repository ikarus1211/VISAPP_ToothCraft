import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import json
import polyscope as ps

def rotate_mesh(vertices, axis, angle_degrees):
    """
    Rotate a set of 3D vertices around a given axis by a given angle (in degrees).

    Parameters:
        vertices (np.ndarray): shape (N, 3)
        axis (str): 'x', 'y', or 'z'
        angle_degrees (float): rotation angle in degrees
    """
    angle = np.radians(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)

    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return vertices @ R.T  # apply rotation

def rotate_json_anot(jaw_json, degree):

    before_flip = []
    after_flip = []
    for fdi, tooth in jaw_json['segmentation'].items():
        before_flip.append(np.asarray(tooth['vertices']).mean(axis=0))
        tooth_vertices = np.array(tooth['vertices'])
        rotated_vertices = rotate_mesh(tooth_vertices, 'x', degree)
        after_flip.append(rotated_vertices.mean(axis=0))
        jaw_json['segmentation'][fdi]['vertices'] = rotated_vertices.tolist()

    return jaw_json


if __name__ == "__main__":
    # Load paths to jaws
    ODD_P = "../New_Teeth/Orthodontic_dental_dataset"
    OUT_P = "../New_Teeth_Flipped/data"
    files = os.listdir(ODD_P)
    files.sort()
    paths = [os.path.join(ODD_P, f) for f in files]
    out_paths = [os.path.join(OUT_P, f) for f in files]


    summ = 0
    to_flip = []
    for idx, path in enumerate(paths):
        mesh_L = trimesh.load(os.path.join(path, "final/L_Final.stl"))
        mesh_U = trimesh.load(os.path.join(path, "final/U_Final.stl"))

        json_L = json.load(open(os.path.join(path, "final/L_Final.json")))
        json_U = json.load(open(os.path.join(path, "final/U_Final.json")))

        mean_vertex_L = np.asarray(mesh_L.vertices).mean(axis=0)
        mean_vertex_U = np.asarray(mesh_U.vertices).mean(axis=0)

        os.mkdir(out_paths[idx])
        os.mkdir(os.path.join(out_paths[idx], "final"))

        if (mean_vertex_L - mean_vertex_U)[2] < 0:
            summ = summ + 1


        else:
            to_flip.append(path)
            rot_L = rotate_mesh(mesh_L.vertices, 'x', 180)
            mesh_L.vertices = rot_L
            rot_U = rotate_mesh(mesh_U.vertices, 'x', 180)
            mesh_U.vertices = rot_U

            json_L = rotate_json_anot(json_L, 180)
            json_U= rotate_json_anot(json_U, 180)

            mean_rot_L = np.asarray(rot_L).mean(axis=0)
            mean_rot_U = np.asarray(rot_U).mean(axis=0)

        mesh_L.export(os.path.join(out_paths[idx], "final/L_Final.stl"))
        mesh_U.export(os.path.join(out_paths[idx], "final/U_Final.stl"))

        json.dump(json_L, open(os.path.join(out_paths[idx], "final/L_Final.json"), "w"))
        json.dump(json_U, open(os.path.join(out_paths[idx], "final/U_Final.json"), "w"))
        print(idx)
