from .blender import BlenderDataset
from .llff import LLFFDataset
from .carla import CarlaDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'carla': CarlaDataset}