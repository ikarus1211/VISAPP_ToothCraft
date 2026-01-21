
import polyscope as ps
import numpy as np

def ps_register_jaw(jaw, teeth_group=None):
    ps.register_surface_mesh("jaw", np.asarray(jaw.mesh_tri.vertices), np.asarray(jaw.mesh_tri.faces))
    ps.register_surface_mesh("gingiva", jaw.gingiva.mesh_tri.vertices, jaw.gingiva.mesh_tri.faces)
    for fdi, tooth in jaw.jaw_teeth.items():
        mesh = ps.register_surface_mesh(fdi, tooth.mesh_tri.vertices, tooth.mesh_tri.faces)
        if teeth_group:
            mesh.add_to_group(teeth_group)

def ps_visualize_jaw(jaw):

    ps.init()
    ps_register_jaw(jaw)
    ps.show()
