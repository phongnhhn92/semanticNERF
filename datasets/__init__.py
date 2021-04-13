from .blender import BlenderDataset
from .llff import LLFFDataset
from .carla import CarlaDataset
from .carla_GVS import CarlaGVSDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'carla': CarlaDataset,
                'carlaGVS': CarlaGVSDataset}