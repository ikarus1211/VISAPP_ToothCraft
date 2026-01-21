import trimesh

from ..utils_module import LocalPath
from .ShapeSpectrum import ShapeSpectrum
from ..geometry_module import MeshGeometry, Trimesh

class TriDVShape:
    def __init__(self,
                 input_data:    LocalPath | MeshGeometry | Trimesh,
                 get_spectrum:  bool = False,
                 ) -> None:
        self.mesh_tri: Trimesh = self.load_geometry(input_data)

        self.spectrum: None | ShapeSpectrum = None
        if get_spectrum:
            self.spectrum = self.load_spectrum()

        # Other representations can be conditionally loaded here...

    @staticmethod
    def load_geometry(input_data: LocalPath | MeshGeometry | Trimesh) -> Trimesh:
        #print(type(input_data))
        if isinstance(input_data, LocalPath):
            return trimesh.load_mesh(input_data, process=False)
        elif type(input_data) == MeshGeometry:
            vertices, faces = input_data
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        elif type(input_data) == Trimesh:
            return input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")

    def load_spectrum(self) -> ShapeSpectrum:
        return ...
