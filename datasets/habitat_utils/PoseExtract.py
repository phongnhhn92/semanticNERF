import numpy as np
import habitat_sim.registry as registry
import matplotlib.pyplot as plt
from habitat_sim.utils.data import ImageExtractor, PoseExtractor
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


@registry.register_pose_extractor(name="random_pose_extractor")
class RandomPoseExtractor(PoseExtractor):
    def extract_poses(self, view, fp):
        height, width = view.shape
        num_random_points = 4
        points = []
        while len(points) < num_random_points:
            # Get the row and column of a random point on the topdown view
            row, col = np.random.randint(0, height), np.random.randint(0, width)

            # Convenient method in the PoseExtractor class to check if a point
            # is navigable
            if self._valid_point(row, col, view):
                points.append((row, col))

        poses = []

        # Now we need to define a "point of interest" which is the point the camera will
        # look at. These two points together define a camera position and angle
        for point in points:
            r, c = point
            point_of_interest = (r - 1, c) # Just look forward
            pose = (point, point_of_interest, fp)
            poses.append(pose)

        return poses

scene_filepath = "/media/phong/Data2TB/dataset/Replica/Replica-Dataset/dataset/apartment_0/habitat/mesh_semantic.ply"
extractor = ImageExtractor(
    scene_filepath,
    img_size=(600,800),
    output=["rgba", "depth", "semantic"],
    pose_extractor_name="panorama_extractor"
)

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

dataset = SemanticSegmentationDataset(extractor,
    transforms=T.Compose([T.ToTensor()])
)

BATCH_SIZE = 4
# Create a Dataloader to batch and shuffle our data
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def show_batch(sample_batch):
    def show_row(imgs, batch_size, img_type):
        plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, batch_size, i + 1)
            ax.axis("off")
            if img_type == 'rgb':
                plt.imshow(img.numpy().transpose(1, 2, 0))
            elif img_type == 'truth':
                plt.imshow(img.numpy())

        plt.show()

    batch_size = len(sample_batch['rgb'])
    for k in sample_batch.keys():
        show_row(sample_batch[k], batch_size, k)


_, sample_batch = next(enumerate(dataloader))
show_batch(sample_batch)