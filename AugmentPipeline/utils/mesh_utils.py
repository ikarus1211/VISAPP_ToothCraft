import numpy as np
import trimesh

def normalize_meshes_for_sdf(meshes, mesh_scale=0.8, inplace=True):

    if not inplace:
        meshes = [mesh.copy() for mesh in meshes]

        # Stack all vertices together to compute global bounds
    all_vertices = np.vstack([mesh.vertices for mesh in meshes])

    bbmin = all_vertices.min(0)
    bbmax = all_vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()

    # Translate meshes to origin
    for mesh in meshes:
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)

    return meshes

def normalize_mesh_for_sdf(mesh, bounds, mesh_scale=0.8, inplace=True):


    all_vertices = np.asarray(mesh.vertices)

    bbmin = all_vertices.min(0)
    bbmax = all_vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()


    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    bounds[0] = (bounds[0] - center) * scale
    bounds[1] = (bounds[1] - center) * scale

    return mesh, np.asarray(bounds)

def normalize_meshes_to_unit_sphere(meshes, inplace=True):
    """
    Normalize a list of trimesh objects to fit within a unit sphere centered at origin.
    This scales and translates each mesh so that the combined set fits in a unit sphere.
    """
    if not inplace:
        meshes = [mesh.copy() for mesh in meshes]

    # Stack all vertices together to compute global bounds
    all_vertices = np.vstack([mesh.vertices for mesh in meshes])

    # Compute centroid of all meshes
    centroid = all_vertices.mean(axis=0)

    # Translate meshes to origin
    for mesh in meshes:
        mesh.vertices -= centroid

    # Recompute all vertices after centering
    all_vertices_centered = np.vstack([mesh.vertices for mesh in meshes])

    # Compute max distance from origin (for unit sphere scaling)
    max_dist = np.linalg.norm(all_vertices_centered, axis=1).max()

    # Scale all meshes
    for mesh in meshes:
        mesh.vertices /= max_dist

    return meshes

def get_primitive_object(object_type, scaled_bbox, size_range=[0.05, 0.10]):
    obj_mesh = None
    bbox_scale =  np.prod(scaled_bbox[1] - scaled_bbox[0]) ** (1/3)
    size_range[0] *= bbox_scale
    size_range[1] *= bbox_scale
    if object_type == "box":
        x_size = np.random.uniform(*size_range)
        y_size = np.random.uniform(*size_range)
        z_size = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.box(extents=[x_size, y_size, z_size])
    elif object_type == "cylinder":
        random_radius = np.random.uniform(*size_range)
        random_height = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.cylinder(
            radius=random_radius, height=random_height
        )
    elif object_type == "cone":
        random_radius = np.random.uniform(*size_range)
        random_height = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.cone(radius=random_radius, height=random_height)
    elif object_type == "capsule":
        random_radius = np.random.uniform(*size_range)
        random_height = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.capsule(
            radius=random_radius, height=random_height
        )
        obj_mesh.apply_scale(0.8)
    elif object_type == "uv_sphere":
        random_radius = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.uv_sphere(radius=random_radius)
    elif object_type == "annulus":
        random_radius = np.random.uniform(*size_range)
        random_height = np.random.uniform(*size_range)
        obj_mesh = trimesh.creation.annulus(
            r_min=random_radius, r_max=2 * random_radius, height=random_height
        )
        obj_mesh.apply_scale(0.6)
    elif object_type == "icosahedron":
        obj_mesh = trimesh.creation.icosahedron()
    else:
        raise NotImplementedError
    return obj_mesh

def random_rotate(mesh):
        mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
        return mesh

def get_translation(mesh, tooth_bounds, shift_mag, shift_dir):

    def shift_bbox(direction, bounds, magnitude):
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        # Extent of the box
        extent = bounds[1] - bounds[0]
        # Project extent onto direction to get how much of the box lies in that direction
        length_along_dir = np.dot(extent, np.abs(direction))  # use abs to get size regardless of sign
        # Compute actual shrink distance
        delta = magnitude * length_along_dir
        # Apply shrink
        new_bounds = bounds.copy()
        for i in range(3):
            if direction[i] < 0:
                new_bounds[1, i] += direction[i] * delta
            elif direction[i] > 0:
                new_bounds[0, i] += direction[i] * delta
        return new_bounds

    # Shift the bbox Y boundary upward so the holes are generated in upper part of tooth
    new_bounds = shift_bbox(shift_dir, tooth_bounds, shift_mag)

        # Iterates until it finds at least one suitable point
    for it in range(1000):
        points, _ = trimesh.sample.sample_surface(mesh, 100)
        points = np.array(points)

        inside_bbox = np.all((points >= new_bounds[0]) &
                             (points <= new_bounds[1]), axis=1)
        points_inside_bbox = points[inside_bbox]
        if points_inside_bbox.shape[0] > 0:
            print(it)
            break

    rand_point_idx = np.random.choice(points_inside_bbox.shape[0])
    rand_point = points_inside_bbox[rand_point_idx]
    return rand_point, new_bounds

def scale_to_box(mesh, aabb, alpha=0.4):

    mesh_bounds = mesh.bounds
    mesh_size = mesh_bounds[1] - mesh_bounds[0]

    # Calculate the size of the target bounding box
    target_size = aabb.get_max_bound() - aabb.get_min_bound()

    # Compute the scaling factor to fit the mesh within the target box
    scaling_factor = min(target_size / mesh_size)

    # Scale the mesh around its centroid
    mesh.apply_scale(scaling_factor * alpha)

    return mesh

def scale_bbox(bbox, scale):

    center = (bbox[0] + bbox[1]) / 2
    half_size = (bbox[1] - bbox[0]) / 2
    scaled_half_size = half_size * scale
    new_min = center - scaled_half_size
    new_max = center + scaled_half_size

    return [new_min, new_max]


