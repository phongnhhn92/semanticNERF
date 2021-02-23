import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from torchvision.transforms import ToTensor

from habitat_sim.utils.data import ImageExtractor
from torchvision import transforms as T

# Replace with the path to your scene file
SCENE_FILEPATH = '/media/phong/data/dataset/replica/apartment_0/mesh.ply'
BATCH_SIZE = 4

class SemanticSegmentationDataset(Dataset):
    def __init__(self, extractor, transforms=None):
        # Define an ImageExtractor
        self.extractor = extractor

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

        # Habitat sim outputs instance id's from the semantic sensor (i.e. two
        # different chairs will be marked with different id's). So we need
        # to create a mapping from these instance id to the class labels we
        # want to predict. We will use the below dictionaries to define a
        # funtion that takes the raw output of the semantic sensor and creates
        # a 2d numpy array of out class labels.
        self.labels = {
            'background': 0,
            'wall': 1,
            'floor': 2,
            'ceiling': 3,
            'chair': 4,
            'table': 5,
        }
        self.instance_id_to_name = self.extractor.instance_id_to_name
        self.map_to_class_labels = np.vectorize(
            lambda x: self.labels.get(self.instance_id_to_name.get(x, 0), 0)
        )

    def __len__(self):
        return len(self.extractor)

    def __getitem__(self, idx):
        sample = self.extractor[idx]
        raw_semantic_output = sample['semantic']
        truth_mask = self.get_class_labels(raw_semantic_output)

        output = {
            'rgb': sample['rgba'][:, :, :3],
            'truth': truth_mask.astype(int),
        }

        if self.transforms:
            output['rgb'] = self.transforms(output['rgb'])
            output['truth'] = self.transforms(output['truth']).squeeze(0)

        return output

    def get_class_labels(self, raw_semantic_output):
        return self.map_to_class_labels(raw_semantic_output)


# extractor = ImageExtractor(SCENE_FILEPATH, output=['rgba', 'semantic'])
#
# dataset = SemanticSegmentationDataset(extractor,
#     transforms=T.Compose([T.ToTensor()])
# )
#
# # Create a Dataloader to batch and shuffle our data
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# def show_batch(sample_batch):
#     def show_row(imgs, batch_size, img_type):
#         plt.figure(figsize=(12, 8))
#         for i, img in enumerate(imgs):
#             ax = plt.subplot(1, batch_size, i + 1)
#             ax.axis("off")
#             if img_type == 'rgb':
#                 plt.imshow(img.numpy().transpose(1, 2, 0))
#             elif img_type == 'truth':
#                 plt.imshow(img.numpy())
#
#         plt.show()
#
#     batch_size = len(sample_batch['rgb'])
#     for k in sample_batch.keys():
#         show_row(sample_batch[k], batch_size, k)
#
#
# _, sample_batch = next(enumerate(dataloader))
# show_batch(sample_batch)