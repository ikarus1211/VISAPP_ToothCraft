import trimesh

from .sdf_utils import *

def jaw_teeth_union(jaw):

    jaw_mesh = jaw.gingiva.mesh_tri
    teeth_mesh = [tooth.mesh_tri for fdi, tooth in jaw.jaw_teeth.items()]
    jaw_union = trimesh.boolean.union(teeth_mesh + [jaw_mesh], engine='manifold')
    return jaw_union

def jaw_teeth_sdf_union(jaw):
    jaw_mesh = jaw.gingiva.mesh_tri
    teeth_mesh = [tooth.mesh_tri for fdi, tooth in jaw.jaw_teeth.items()]
    jaw_union = []

    # Convert jaw to sdf
    jaw_sdf = mesh_to_sdf(jaw_mesh, 512)

    for tooth_m in teeth_mesh:
        tooth_sdf = mesh_to_sdf(tooth_m, 512)
        jaw_sdf = sdf_union(jaw_sdf, tooth_sdf)

    return jaw_union

def slice_mesh_ood(m, bbox, jaw_type, cap=True):

    if jaw_type == "lower":
        #
        m = trimesh.intersections.slice_mesh_plane(m, [0, 0, -1.0], bbox[1], cap=cap)

        # Front Back
        m = trimesh.intersections.slice_mesh_plane(m, [0, 1.0, 0], bbox[0], cap=cap)
        m = trimesh.intersections.slice_mesh_plane(m, [0, -1.0, 0], bbox[1], cap=cap)

        # Left Right
        m = trimesh.intersections.slice_mesh_plane(m, [1.0, 0, 0], bbox[0], cap=cap)
        m = trimesh.intersections.slice_mesh_plane(m, [-1.0, 0, 0], bbox[1], cap=cap)
    elif jaw_type == "upper":
        #
        m = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1.0], bbox[0], cap=cap)

        # Front Back
        m = trimesh.intersections.slice_mesh_plane(m, [0, 1.0, 0], bbox[0], cap=cap)
        m = trimesh.intersections.slice_mesh_plane(m, [0, -1.0, 0], bbox[1], cap=cap)

        # Left Right
        m = trimesh.intersections.slice_mesh_plane(m, [1.0, 0, 0], bbox[0], cap=cap)
        m = trimesh.intersections.slice_mesh_plane(m, [-1.0, 0, 0], bbox[1], cap=cap)
    else:
        raise TypeError("jaw_type must be 'lower' or 'upper'")

    return m

def is_booleanable(mesh):
    cube = trimesh.creation.box(extents=mesh.extents)

    try:
        result = mesh.intersection(cube)
        print("Boolean operation succeeded.")
    except Exception as e:
        print("Boolean operation failed:", e)



