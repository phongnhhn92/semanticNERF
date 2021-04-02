import numpy as np
import matplotlib.pyplot as plt

from habitat_sim.utils.data import ImageExtractor


# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

#/media/phong/Data2TB/dataset/Replica/Replica-Dataset/dataset/apartment_0
#scene_filepath = "/home/phong/data/Work/Paper4/Library/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
scene_filepath = "/media/phong/Data2TB/dataset/Replica/Replica-Dataset/dataset/apartment_0/habitat/mesh_semantic.ply"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('train')

# Index in to the extractor like a normal python list
sample = extractor[0]

# Or use slicing
samples = extractor[1:4]
for sample in samples:
    display_sample(sample)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()