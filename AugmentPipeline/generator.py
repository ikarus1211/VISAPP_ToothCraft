import logging
import os
import os.path as osp
import polyscope as ps
import skimage
import trimesh.boolean
import random
import string
from pathlib import Path

from matplotlib.patches import draw_bbox

from ShapeLib.data_module import JawSample

from utils.bool_utils import jaw_teeth_union, is_booleanable, slice_mesh_ood
from utils.sdf_utils import *
from utils.mesh_utils import *
from utils.misc import Timer

class Generator:
    def __init__(self, config):
        self.config = config
        self.data_name = config.data.name
        self.object_list = (
            [
                "box",
                "cylinder",
                "cone",
                "capsule",
                "uv_sphere",
            ]
        )
        self.save_token = self.generate_token()
        logging.basicConfig(level=logging.CRITICAL)

        self.l_proc_time = Timer()
        self.u_proc_time = Timer()
        self.fdis = []
        self.ious = []
        self.zero_ious = []

    def generate_token(self):
        letter = random.choice(string.ascii_letters)  # Random letter (uppercase or lowercase)
        number1 = random.choice(string.digits)  # Random digit
        number2 = random.choice(string.digits)  # Another random digit
        return f"{letter}{number1}{number2}"

    def convert_teeth(self, jaw):


        for fdi, tooth in jaw.jaw_teeth.items():
            # MUST be normalized
            center = tooth.mesh_tri.centroid
            # Returns normalized mesh together with
            norm_tooth, _ = normalize_trimesh(tooth.mesh_tri.copy(), 0.99)
            tooth_sdf = mesh_to_sdf(norm_tooth, 32)
            # Map it back to original position
            tooth_sdf = map_voxel_grid_with_translation(tooth_sdf,
                                                        np.zeros((256,256,256)),
                                                        -norm_tooth.centroid, [[-1,-1,-1],[1,1,1]])

            vertices, faces, _, _ = skimage.measure.marching_cubes(tooth_sdf, level=0.0)
            #ps.register_surface_mesh(f"{fdi}_r", vertices, faces)


    def sdf_jaw_union(self, jaw):
        pass


    def get_comp_slices(self, jaw, fdi, tooth, bounds, jaw_type):

        # Get Gingiva and tooth meshes
        gin = jaw.gingiva.mesh_tri.copy()

        # Slice of the surrounding gaging without teeth
        slice_gin = slice_mesh_ood(gin.copy(), bounds, jaw_type)
        mesh_components = [slice_gin]

        # Get closest neighbours
        left_idx, right_idx = jaw.get_fdi_neighbours(fdi)
        # Slice available neighbours
        if left_idx != 0:
            left_tooth = jaw.jaw_teeth[left_idx].mesh_tri.copy()
            left_slice = slice_mesh_ood(left_tooth, bounds, jaw_type)
            mesh_components.append(left_slice)
        if right_idx != 0:
            right_tooth = jaw.jaw_teeth[right_idx].mesh_tri.copy()
            right_slice = slice_mesh_ood(right_tooth, bounds, jaw_type)
            mesh_components.append(right_slice)

        # Append the Center tooth
        mesh_components.append(tooth.mesh_tri)

        return mesh_components

    def components_to_sdf(self, components):
        # Transform to sdf
        sdf_slices = []
        res = self.config.gen.resolution
        for mesh in components:
            sdf = mesh_to_sdf(mesh, res)
            sdf_slices.append(sdf)

        return sdf_slices

    def make_holes_3ds(self, tooth_sdf, tooth_mesh, res, bbox):

            for i in range(0, random.randint(1, self.config.gen.max_num_holes)):
                object_type = np.random.choice(self.object_list)
                obj_mesh = get_primitive_object(object_type, list(self.config.gen.size))
                # First rotate the object
                obj_mesh = random_rotate(obj_mesh)
                # Scale to fit inside top half of bounding box
                # Translate the obj to random point on tooth surface
                # self.draw_bbox(tooth_mesh.bounds, f"a_{fdi}")
                rand_point, new_bounds = get_translation(tooth_mesh, bbox,
                                                  self.config.gen.shift_mag, [0, 0, -1])
                #self.draw_bbox(new_bounds, f"b_")
                obj_mesh.apply_translation(rand_point)

                obj_mesh = slice_mesh_ood(obj_mesh, bbox, 'upper')

                obj_sdf = mesh_to_sdf(obj_mesh, res)
                # Noise Addition
                if hasattr(self.config, 'noise') and self.config.noise.is_on:
                    obj_sdf, _ = add_noise_to_sdf(obj_sdf, amp=self.config.noise.amp, freq=self.config.noise.freq)

                tooth_sdf = sdf_difference(tooth_sdf, obj_sdf)

            return tooth_sdf


    def makes_holes(self, tooth_sdf, tooth_mesh, res, fdi=0, type='lower', bbox=None):

        for i in range(0, random.randint(1, self.config.gen.max_num_holes)):
            object_type = np.random.choice(self.object_list)
            scaled_bbox = scale_bbox(bbox, self.config.gen.bbox_scale)
            obj_mesh = get_primitive_object(object_type, scaled_bbox, list(self.config.gen.size))
            # First rotate the object
            obj_mesh = random_rotate(obj_mesh)
            # Scale to fit inside top half of bounding box
            # Translate the obj to random point on tooth surface
            # self.draw_bbox(tooth_mesh.bounds, f"a_{fdi}")
            if type == 'lower':
                rand_point, box = get_translation(tooth_mesh, tooth_mesh.bounds,
                                                  self.config.gen.shift_mag, [0, 0, 1])
            else:
                rand_point, box = get_translation(tooth_mesh, tooth_mesh.bounds,
                                                  self.config.gen.shift_mag, [0, 0, -1])

            # Register the point cloud
            #ps.register_point_cloud(f"p_{i}", np.array([rand_point]))

            #  self.draw_bbox(box, f"b")

            obj_mesh.apply_translation(rand_point)

            if bbox is not None:
                obj_mesh = slice_mesh_ood(obj_mesh, bbox, type)

            obj_sdf = mesh_to_sdf(obj_mesh, res)
            # Noise Addition
            if hasattr(self.config, 'noise') and self.config.noise.is_on:
                obj_sdf, _ = add_noise_to_sdf(obj_sdf, amp=self.config.noise.amp, freq=self.config.noise.freq)

            if self.config.debug:
                vol = ps.register_volume_grid(f"{i}_prim", (res, res, res), (-1, -1, -1), (1, 1, 1))
                vol.add_scalar_quantity("node", obj_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)

            tooth_sdf = sdf_difference(tooth_sdf, obj_sdf)
        if self.config.debug:
            r_isdf = ps.register_volume_grid(f"tooth_{type}", (64, 64, 64), (-1, -1, -1,), (1, 1, 1))
            r_isdf.add_scalar_quantity("vals3", tooth_sdf)
        return tooth_sdf

    def save_models(self, models, name, fdi):
        os.makedirs(osp.join(self.config.data.save, name), exist_ok=True)

        for i,m in enumerate(models):
            m.export(osp.join(osp.join(self.config.data.save, name),  f'{fdi}_{i}.obj'))



    def procces_ood_jaw_separately(self, jaw, type):
        """ Produces cases with only one jaw present,
            Input is a jaw and "lower" or "upper" for jaw type
        """


        if self.config.debug:
            ps.init()
        if jaw.gingiva == -1 or jaw.jaw_teeth == -1:
            return
        # ps.register_surface_mesh("Gingiva", jaw.gingiva.mesh_tri.vertices, jaw.gingiva.mesh_tri.faces)
        # for k,v in jaw.jaw_teeth.items():
        #     ps.register_surface_mesh(f"{k}", v.mesh_tri.vertices, v.mesh_tri.faces)
        # for k,v in jaw.jaw_teeth.items():
        #     self.draw_bbox(scale_bbox(v.mesh_tri.bounds , 2) , f"bounds {k}")
        # ps.show()
        # ps.remove_all_structures()

        for fdi, tooth in jaw.jaw_teeth.items():
            print(fdi)
            tooth_center = tooth.mesh_tri.centroid

            # Debug
            trim_tooth = tooth.mesh_tri.copy()
            scaled_bbox = trim_tooth.apply_translation(-tooth_center).apply_scale(
                self.config.gen.bbox_scale).apply_translation(tooth_center).bounds

            # Get mesh components
            mesh_components = self.get_comp_slices(jaw, fdi, tooth, scaled_bbox, type)

            # Normalize the components together for SDF creation
            normalized_meshes = normalize_meshes_to_unit_sphere(mesh_components, False)
            normalized_for_sdf = normalize_meshes_for_sdf(normalized_meshes, 0.8, False)

            if hasattr(self.config, 'save_models') and self.config.save_models:
                self.save_models(normalized_for_sdf, jaw.sample_id.split(".")[0], str(fdi))

            # Bounds for Metric calculation
            bounds = normalized_for_sdf[-1].bounds

            del normalized_meshes
            sdf_slices = self.components_to_sdf(normalized_for_sdf)

            # Create holes in tooth
            hole_teeth = self.makes_holes(sdf_slices[-1].copy(), normalized_for_sdf[-1].copy(),
                                              res=self.config.gen.resolution, fdi=fdi, type=type, bbox=bounds)
            del normalized_for_sdf
            # Union of the teeth and gingiva
            gt_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for sdf_slice in sdf_slices:
                gt_sdf = sdf_union(sdf_slice, gt_sdf)

            sdf_slices[-1] = hole_teeth
            cond_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for sdf_slice in sdf_slices:
                cond_sdf = sdf_union(sdf_slice, cond_sdf)

            # Rectify the SDF error
            vertices, faces, _, _ = skimage.measure.marching_cubes(cond_sdf, level=0.0)
            normalized_for_sdf = normalize_meshes_for_sdf([trimesh.Trimesh(vertices, faces)], 0.8, False)[0]
            cond_sdf = mesh2sdf.compute(normalized_for_sdf.vertices, normalized_for_sdf.faces, size=self.config.gen.resolution)


            self.ious.append(self.compute_iou_sdf(gt_sdf, cond_sdf, self.get_bound_mask(cond_sdf, bounds)))
            zero_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for slice in sdf_slices[:-1]:
                zero_sdf = sdf_union(slice, zero_sdf)
            self.zero_ious.append(self.compute_iou_sdf(gt_sdf, zero_sdf, self.get_bound_mask(zero_sdf, bounds)))
            self.fdis.append(fdi)

            if self.config.debug:
                res = self.config.gen.resolution
                vol = ps.register_volume_grid(f"{fdi}_hole", (res, res, res), (-1, -1, -1), (1, 1, 1))
                vol.add_scalar_quantity("node", cond_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)

                gt = ps.register_volume_grid(f"{fdi}_gt", (res, res, res), (-1, -1, -1), (1, 1, 1))
                gt.add_scalar_quantity("node", gt_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)

                ps.show()
                ps.remove_all_structures()

            self.save_sample(cond_sdf, gt_sdf, jaw.sample_id.split(".")[0], str(fdi), bounds=bounds)


    def get_opsing(self, jaw, fdi):
        pass
    def save_sample(self, tsdf, tsdf_gt, model_name, label, antag=None, bounds=None):
        """
        Saves ground truth
        :param tsdf: model in sdf representation
        :param model_name: save name of the model
        :return:
        """
        # Save Bounds

        save_path = os.path.join(
            self.config.data.save, f"{self.config.gen.resolution}", "bounds",
            f"{self.save_token}_{model_name}_b_{label}.npy"
        )
        np.save(save_path, bounds)
        # Save GT
        save_path = os.path.join(
            self.config.data.save, f"{self.config.gen.resolution}", "gt", f"{self.save_token}_{model_name}_gt_{label}.npy"
        )
        np.save(save_path, tsdf_gt)

        # Save sample
        save_path = os.path.join(
            self.config.data.save, f"{self.config.gen.resolution}", "incomplete" ,f"{self.save_token}_{model_name}_{label}.npy"
        )
        np.save(save_path, tsdf)
        # Save antagonist
        if antag is not None:
            save_path = os.path.join(
                self.config.data.save, f"{self.config.gen.resolution}", "antag",
                f"{self.save_token}_{model_name}_antag_{label}.npy"
            )
            np.save(save_path, antag)

    def get_antag_comp(self, antag_jaw, bounds, type):

        mesh_components = []
        for fdi, tooth in antag_jaw.jaw_teeth.items():
            trim_tooth = tooth.mesh_tri.copy()
            tooth_slice = slice_mesh_ood(trim_tooth, bounds, type)
            mesh_components.append(tooth_slice)
        return mesh_components


    def proces_jaw_jointly(self, jaw, antag_jaw, type='lower'):


        if jaw.gingiva == -1 or jaw.jaw_teeth == -1:
            return
        
        if antag_jaw.gingiva == -1 or antag_jaw.jaw_teeth == -1:
            return
        if self.config.debug:
            ps.init()
        # ps.register_surface_mesh(f"Gingiva_{type}", jaw.gingiva.mesh_tri.vertices, jaw.gingiva.mesh_tri.faces)
        # for k,v in jaw.jaw_teeth.items():
        #     ps.register_surface_mesh(f"{k}", v.mesh_tri.vertices, v.mesh_tri.faces)
        # for k,v in jaw.jaw_teeth.items():
        #     self.draw_bbox(scale_bbox(v.mesh_tri.bounds , 2) , f"bounds {k}")
        # ps.show()
        # ps.remove_all_structures()

        for fdi, tooth in jaw.jaw_teeth.items():
            print(fdi)
            tooth_center = tooth.mesh_tri.centroid
            # Debug
            trim_tooth = tooth.mesh_tri.copy()
            scaled_bbox = trim_tooth.apply_translation(-tooth_center).apply_scale(
                self.config.gen.bbox_scale).apply_translation(tooth_center).bounds

            # Get mesh components
            mesh_components = self.get_comp_slices(jaw, fdi, tooth, scaled_bbox, type)


            if type == 'lower':
                antag_comp = self.get_antag_comp(antag_jaw, scaled_bbox, 'upper')
            else:
                antag_comp = self.get_antag_comp(antag_jaw, scaled_bbox, 'lower')

            if self.config.antag.method == 'separate':
                normalized_for_sdf = normalize_meshes_for_sdf(mesh_components + antag_comp, 0.8, False)
                # Bounds for Metric calculation
                bounds = normalized_for_sdf[- (len(antag_comp) + 1)].bounds

                combined_sdf = self.components_to_sdf(normalized_for_sdf)
                sdf_slices = combined_sdf[:len(mesh_components)]
                sdf_antag = combined_sdf[len(mesh_components):]
                del combined_sdf

            elif self.config.antag.method == 'together':
                normalized_for_sdf = normalize_meshes_for_sdf(mesh_components + antag_comp, 0.8, False)
                sdf_slices = self.components_to_sdf(normalized_for_sdf)

            # Create holes in tooth
            hole_teeth = self.makes_holes(sdf_slices[-1].copy(), normalized_for_sdf[:len(mesh_components)][-1].copy(),
                                          res=self.config.gen.resolution, bbox=bounds, fdi=fdi, type=type)
            del normalized_for_sdf
            # GT Union of the teeth and gingiva
            gt_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for sdf_slice in sdf_slices:
                gt_sdf = sdf_union(sdf_slice, gt_sdf)

            # Sample Union of different components
            sdf_slices[-1] = hole_teeth
            cond_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for sdf_slice in sdf_slices:
                cond_sdf = sdf_union(sdf_slice, cond_sdf)

            # Rectify the SDF error
            res = self.config.gen.resolution
            spacing = (2 / (res - 1), 2 / (res - 1), 2 / (res - 1))
            vertices, faces, _, _ = skimage.measure.marching_cubes(cond_sdf, level=0.0, spacing=spacing)
            vertices = vertices + np.array([-1.0, -1.0, -1.0])
            #normalized_for_sdf = normalize_meshes_for_sdf([trimesh.Trimesh(vertices, faces),], 0.8, False)[0]
            cond_sdf = mesh2sdf.compute(vertices, faces, size=self.config.gen.resolution)

            self.ious.append(self.compute_iou_sdf(gt_sdf, cond_sdf, self.get_bound_mask(cond_sdf, bounds)))
            self.fdis.append(fdi)
            zero_sdf = np.ones_like(sdf_slices[0]) * np.inf
            for slice in sdf_slices[:-1]:
                zero_sdf = sdf_union(slice, zero_sdf)
            self.zero_ious.append(self.compute_iou_sdf(gt_sdf, zero_sdf, self.get_bound_mask(zero_sdf, bounds)))

            if self.config.debug:
                res = self.config.gen.resolution
                vol = ps.register_volume_grid(f"{fdi}_hole", (res, res, res), (-1, -1, -1), (1, 1, 1))
                vol.add_scalar_quantity("node", cond_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)

                gt = ps.register_volume_grid(f"{fdi}_gt", (res, res, res), (-1, -1, -1), (1, 1, 1))
                gt.add_scalar_quantity("node", gt_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)

            # Antag union of different components
            antag_sdf = np.ones_like(sdf_antag[0]) * np.inf
            for antag in sdf_antag:
                antag_sdf = sdf_union(antag, antag_sdf)

            self.save_sample(cond_sdf, gt_sdf, jaw.sample_id.split(".")[0], str(fdi), antag=antag_sdf, bounds=bounds)

            if self.config.debug:
                res = self.config.gen.resolution
                vol = ps.register_volume_grid(f"{fdi}_r", (res, res, res), (-1, -1, -1), (1, 1, 1))
                vol.add_scalar_quantity("node", antag_sdf, defined_on='nodes', enabled=True, enable_isosurface_viz=True)


                ps.show()
                ps.remove_all_structures()


    def process_ood(self, l_jaw, u_jaw, antagonist=True):

        #ps.init()

        if antagonist:
            self.proces_jaw_jointly(l_jaw, u_jaw, 'lower')
            self.proces_jaw_jointly(u_jaw, l_jaw, 'upper')
        else:
            self.l_proc_time.tic()
            # Process lower
            self.procces_ood_jaw_separately(l_jaw, "lower")
            logging.info(self.l_proc_time.toc())

            self.u_proc_time.tic()
            # Process Upper
            self.procces_ood_jaw_separately(u_jaw, "upper")
            logging.info(self.u_proc_time.toc())

    def process_3ds_separately(self, jaw, type):
        labels = jaw.jaw_json['labels']
        unique_labels = list(set(labels))

        if self.config.debug:
            print('psinit') # ps.init()

        for fdi in unique_labels:
            if fdi == 0:
                continue

            seg_mask = np.asarray(labels, dtype=np.int8) == int(fdi)
            tooth_verts = np.asarray(jaw.mesh_tri.vertices)[seg_mask]
            bounds = [tooth_verts.min(axis=0), tooth_verts.max(axis=0)]
            scaled_bbox = scale_bbox(bounds.copy(), self.config.gen.bbox_scale)

            segmented_mesh = jaw.mesh_tri.copy()

            sliced_mesh = slice_mesh_ood(segmented_mesh, scaled_bbox, 'upper')

            if not sliced_mesh.is_watertight:
                logging.info(f"Unable to create hole for model  label {fdi}")
                continue

            normalized_for_sdf, new_bounds = normalize_mesh_for_sdf(sliced_mesh, bounds,0.8, False)

            mesh_sdf = mesh_to_sdf(normalized_for_sdf, self.config.gen.resolution)

            gt_sdf = mesh_sdf.copy()
            cond_sdf = self.make_holes_3ds(mesh_sdf, normalized_for_sdf, self.config.gen.resolution, new_bounds)

            # Rectify the SDF error
            vertices, faces, _, _ = skimage.measure.marching_cubes(cond_sdf, level=0.0)
            normalized_for_sdf =normalize_meshes_for_sdf([trimesh.Trimesh(vertices, faces)], 0.8, False)[0]
            cond_sdf = mesh2sdf.compute(normalized_for_sdf.vertices, normalized_for_sdf.faces, size=self.config.gen.resolution)


            self.ious.append(self.compute_iou_sdf(gt_sdf, cond_sdf))
            self.fdis.append(fdi)
            if self.config.debug:
                cond = ps.register_volume_grid("Cond", (64,64,64), (-1.0,-1.0,-1.0), (1.0,1.0,1.0))
                cond.add_scalar_quantity("Sdf", cond_sdf)
                cond = ps.register_volume_grid("GT", (64, 64, 64), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
                cond.add_scalar_quantity("Sdf", gt_sdf)
                self.draw_bbox(new_bounds, 'new_bounds')
                ps.show()
            else:
                self.save_sample(cond_sdf, gt_sdf, jaw.sample_id, str(fdi), bounds=bounds)

    
    def get_bound_mask(self, sdf, bound, mask_value=1):
        D, H, W = sdf.shape
    
        min_bound, max_bound = bound
    
        # Create voxel grid coordinates in [-1, 1] for each axis
        z = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')  # shape (D, H, W)
    
        coords = np.stack([zz, yy, xx], axis=-1)  # shape (D, H, W, 3)
    
        # Create boolean mask: True where inside bounds
        in_bounds = np.all((coords >= min_bound) & (coords <= max_bound), axis=-1)
    
        # Apply mask
        masked_sdf = np.where(in_bounds, mask_value, 0)
        return masked_sdf
      
    def compute_iou_sdf(self, sdf_a, sdf_b, mask=None):
        """
        This is voxel based IOU so it has precision limited by voxel grid
        """
        inside_a = sdf_a <= 0
        inside_b = sdf_b <= 0

        if mask is not None:
            intersection = np.sum((inside_a & inside_b) * mask)
            union = np.sum((inside_a | inside_b) * mask)
        else:
            intersection = np.sum(inside_a & inside_b)
            union = np.sum(inside_a | inside_b)

        iou = intersection / union if union > 0 else 0.0
        return iou

    def process_3ds(self, l_jaw, u_jaw):

        if l_jaw is not None:
            self.l_proc_time.tic()
            self.process_3ds_separately(l_jaw, "lower")
            logging.info(self.l_proc_time.toc())
        if u_jaw is not None:
            self.u_proc_time.tic()
            self.process_3ds_separately(u_jaw, "upper")
            logging.info(self.u_proc_time.toc())

    def create_folder_structure(self):
        os.makedirs(self.config.data.save, exist_ok=True)
        os.makedirs(osp.join(self.config.data.save, f"{self.config.gen.resolution}"), exist_ok=True)
        os.makedirs(osp.join(self.config.data.save, f"{self.config.gen.resolution}", "gt"), exist_ok=True)
        os.makedirs(osp.join(self.config.data.save, f"{self.config.gen.resolution}", "incomplete"), exist_ok=True)
        os.makedirs(osp.join(self.config.data.save, f"{self.config.gen.resolution}", "bounds"), exist_ok=True)
        if self.config.antag.antagonist:
            os.makedirs(osp.join(self.config.data.save, f"{self.config.gen.resolution}", "antag"), exist_ok=True)

    def check_mesh(self, mesh):
        """
        Check if the mesh after difference operation is empty
        """
        obj_mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        obj_mesh.apply_translation([0, 0, 0])
        try:
            diff_mesh = mesh.difference(obj_mesh, check_volume=False)
        except:
            return True
        return False


    def process_model(self, jaw_paths):
        """
        Takes care of processing model
        :param model_path: path to single model file
        :return:
        """
        if self.data_name == 'OOD':
            id_l = Path(jaw_paths['lower']).parts[-3] + "_" +  Path(jaw_paths['lower']).parts[-1]
            id_u = Path(jaw_paths['upper']).parts[-3] + "_" +  Path(jaw_paths['upper']).parts[-1]
            logging.info(f"Processing model {id_l} and {id_u}")

            l_jaw = JawSample.JawSample(id_l, self.data_name, jaw_paths['lower'], jaw_paths['lower'])
            u_jaw = JawSample.JawSample(id_u, self.data_name, jaw_paths['upper'], jaw_paths['upper'])


            self.process_ood(l_jaw, u_jaw, self.config.antag.antagonist)
            return [self.ious, self.fdis, self.zero_ious]

        elif self.data_name == '3DS':

            if 'lower' in jaw_paths:
                id_l = Path(jaw_paths['lower']).parts[-1].split("_")[0]
                logging.info(f"Processing model {id_l}")
                l_jaw = JawSample.JawSample(id_l, self.data_name, jaw_paths['lower'], jaw_paths['lower'], parse_teeth=False)
            else:
                l_jaw = None
            if 'upper' in jaw_paths:
                id_u = Path(jaw_paths['upper']).parts[-1].split("_")[0]
                logging.info(f"Processing model {id_u}")
                u_jaw = JawSample.JawSample(id_u, self.data_name, jaw_paths['upper'], jaw_paths['upper'], parse_teeth=False)
            else:
                u_jaw = None
            self.process_3ds(l_jaw, u_jaw)



    def draw_bbox(self, bounds, name):

        min_corner, max_corner = bounds

        # Create 8 corner points of the bounding box
        bbox_points = np.array([
            [min_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
        ])

        # Define edges between those 8 points (12 edges)
        bbox_edges = np.array([
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5],
            [2, 3], [2, 6],
            [3, 7],
            [4, 5], [4, 6],
            [5, 7],
            [6, 7]
        ])

        # Register the bounding box as a curve network
        ps.register_curve_network(name, nodes=bbox_points, edges=bbox_edges, radius=0.002)