import os
import json

import numpy as np
import trimesh

from ..utils_module import LocalPath
from .TriDVShape import TriDVShape
from ..geometry_module import MeshGeometry, Trimesh, jaw_parser
from ..data_module import AVAILABLE_JAW_SOURCES, DatasetName, ToothInfo, JawInfo, ToothLabelFDI


class JawSample(TriDVShape):
    def __init__(self,
                 sample_id:     str,
                 dataset_name:  DatasetName,
                 root_path:     LocalPath,
                 input_data:    LocalPath | MeshGeometry | Trimesh,
                 get_spectrum:  bool = False,
                 parse_teeth: bool = True,
                 ) -> None:
        assert dataset_name in AVAILABLE_JAW_SOURCES, f'Unsupported dataset: {dataset_name}.'

        super().__init__(input_data, get_spectrum)

        self.sample_id: str             = sample_id
        self.dataset_name: DatasetName  = dataset_name
        self.root_path: LocalPath       = root_path

        self.jaw_info: dict[JawInfo] = self.load_jaw_info(root_path, dataset_name)
        self.teeth_info: dict[ToothInfo] = self.load_teeth_info(root_path, dataset_name)

        # Stores all separate teeth
        self.jaw_json = self.load_jaw_json(root_path, dataset_name)
        # Initialize all the separate teeth and gingiva
        self.jaw_teeth: dict[ToothLabelFDI, TriDVShape]
        self.gingiva: TriDVShape
        if parse_teeth:
            self.mesh_tri.merge_vertices()
            self.jaw_teeth, self.gingiva = self.load_jaw_teeth(dataset_name)


    def load_jaw_info(self,
                      root_path:    LocalPath,
                      dataset_name: DatasetName,
                      ) -> dict[JawInfo]:
        """
        Method loads information about the jaw from a dataset. Based on the dataset_name,
        it loads the information from the corresponding source into a canonical format.
        tbd: finish docs

        :param root_path:
        :param dataset_name:
        :return:
        """




        return ...

    def load_teeth_info(self,
                        root_path:    LocalPath,
                        dataset_name: DatasetName,
                        ) -> dict[ToothInfo]:
        """
        Method loads information about teeth from a dataset. Based on the dataset_name,
        it loads the information from the corresponding source into a canonical format.

        tbd: finish docs

        :param root_path:
        :param dataset_name:
        :return:
        """
        return ...

    def load_jaw_teeth(self,
                       dataset_name: DatasetName
                       ) -> dict[ToothLabelFDI,TriDVShape]:

        if dataset_name == "OOD":
            return self.get_odd_teeth()
        elif dataset_name == "3DS":
            return self.get_3ds_teeth()
        else:
            raise ValueError("Loading teeth for this dataset is not supported.")

    def get_3ds_teeth(self) -> dict[ToothLabelFDI, TriDVShape]:
        if not "labels" in self.jaw_json:
            raise ValueError("No segmentation information for this dataset.")


    def get_odd_teeth(self)-> (dict[ToothLabelFDI, TriDVShape], TriDVShape):

        if not "segmentation" in self.jaw_json:
            raise ValueError("No segmentation information for this dataset.")


        # Compute centre of mass for each segmented tooth
        seg_com = jaw_parser.com_of_segmentation(self.jaw_json["segmentation"])
        # Split the mesh using CCA
        split_mesh = trimesh.graph.split(self.mesh_tri, only_watertight=False, engine='networkx')
        # Compute the center of mass in CCA
        split_com = jaw_parser.com_of_cca(split_mesh)

        # For saving which of the split meshes is gingiva
        gingiva = list(range(len(split_com)))

        # Computes the minimal distance between centers of masses
        fdi_teeth = {}
        # Check if split components are the same size + 1 than segments
        if len(split_mesh) > len(seg_com) + 1:
            print(f"Cannot parse teeth. Labels: {len(seg_com)}, Components: {len(split_com)} ")
            print(f"Cannot parse teeth. Labels: {len(seg_com)}, Components: {len(split_com)} ")
            return -1, -1
        elif len(split_mesh) < len(seg_com) + 1:
            print(f"Cannot parse teeth. Labels: {len(seg_com)}, Components: {len(split_com)} ")
            return -1, -1



        for label, center in seg_com.items():
            # Find the min dist for this CoM
            distances = np.linalg.norm(split_com - center, axis=1)
            closest_mesh_idx = np.argmin(distances)
            # Save the mesh with corresponding FDI label
            closest_mesh = split_mesh[closest_mesh_idx]
            fdi_teeth[label] = TriDVShape(closest_mesh)
            # Remove this index from gingiva list
            gingiva.remove(closest_mesh_idx)

        if len(gingiva) == 1:
            gingiva_shape = TriDVShape(split_mesh[gingiva[0]])
            return fdi_teeth, gingiva_shape
        else:
            print("Gingiva teeth not found.")
            return fdi_teeth, None

    def get_fdi_neighbours(self, tooth_fdi) -> (int, int):
        """

        Args:
            tooth_fdi: Fdi of the tooth which neighbours are to be found. If tooth does not have a neighbour 0 returned


        Returns:
            left, right: fdi of the neighbours of the tooth
        """

        # Define full arch in anatomical order (upper and lower)

        upper_order = ['18', '17', '16', '15', '14', '13', '12', '11', '21', '22', '23', '24', '25', '26', '27',
                       '28']  # upper (right to left)
        lower_order = ['48', '47', '46', '45', '44', '43', '42', '41', '31', '32', '33', '34', '35', '36', '37',
                       '38']  # lower (left to right)

        # Filter arch to only existing teeth
        if tooth_fdi in upper_order:
            idx = upper_order.index(tooth_fdi)
            search_field = upper_order
        elif tooth_fdi in lower_order:
            idx = lower_order.index(tooth_fdi)
            search_field = lower_order
        else:
            raise ValueError(f"Tooth: {tooth_fdi} not in jaw")

        left, right = 0, 0
        # Left neighbour
        for seatch_i in range(idx + 1, len(search_field)):
            if search_field[seatch_i] in self.jaw_teeth.keys():
                left = search_field[seatch_i]
                break
        # Right neighbour
        for seatch_i in range(idx - 1, 0, -1):
            if search_field[seatch_i] in self.jaw_teeth.keys():
                right = search_field[seatch_i]
                break
        return (left, right)



    def load_jaw_json(self,
                      root_path: LocalPath,
                      dataset_name: DatasetName
                      ):
        """
        Method loads information about the jaw from a dataset. Based on the dataset_name,
        :param root_path:
        :param dataset_name:
        :return:
        """
        # TODO Not robust
        if dataset_name == "OOD":
            json_path = os.path.splitext(root_path)[0] + ".json"
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            return json_data
        elif dataset_name == "3DS":
            json_path = os.path.splitext(root_path)[0] + ".json"
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            return json_data
        else:
            ValueError("Loading json for this model is not supported.")
