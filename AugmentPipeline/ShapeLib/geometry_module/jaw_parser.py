import numpy as np

def com_of_segmentation(seg_vertices):

    seg_com = {}
    for key, val in seg_vertices.items():
        # Returns a mesh of a tooth in trimesh
        vertices = np.asarray(val["vertices"])
        com = np.mean(vertices, axis=0)
        seg_com[key] = com
    return seg_com


def com_of_cca(cca_meshes):

    split_com = []
    for idx, mesh in enumerate(cca_meshes):
        com = np.mean(mesh.vertices, axis=0)
        split_com.append(com)
    return np.array(split_com)
