# [setup]
import math
import os

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from matplotlib import pyplot as plt
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "semantic_object_tutorial_output/")

save_index = 0


def show_img(data, save):
    # display rgb and semantic images side-by-side
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.imshow(data[0], interpolation="nearest")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.imshow(data[1], interpolation="nearest")
    plt.axis("off")
    plt.show(block=False)
    if save:
        global save_index
        plt.savefig(
            output_path + str(save_index) + ".jpg",
            bbox_inches="tight",
            pad_inches=0,
            quality=50,
        )
        save_index += 1
    plt.pause(1)


def get_obs(sim, show, save):
    # render sensor ouputs and optionally show them
    rgb_obs = sim.get_sensor_observations()["rgba_camera"]
    semantic_obs = sim.get_sensor_observations()["semantic_camera"]

    rgb = rgb_obs[...,:3]
    img = Image.fromarray(rgb)
    img.save(os.path.join(output_path,'rgb.png'))
    # if show:
    #     show_img((rgb_obs, semantic_obs), save)


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [1.0, 0.0, 1.0]
    agent_state.rotation = quat_from_angle_axis(
        math.radians(70), np.array([0, 1.0, 0])
    ) * quat_from_angle_axis(math.radians(-20), np.array([1.0, 0, 0]))
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration(scene_file):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup rgb and semantic sensors
    camera_resolution = [600, 800]
    sensor_specs = []

    rgba_camera_spec = habitat_sim.SensorSpec()
    rgba_camera_spec.uuid = "rgba_camera"
    rgba_camera_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_spec.resolution = camera_resolution
    rgba_camera_spec.postition = [0.0, 0.0, 0.0]  # ::: fix y to be 0 later
    rgba_camera_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_spec)

    semantic_camera_spec = habitat_sim.SensorSpec()
    semantic_camera_spec.uuid = "semantic_camera"
    semantic_camera_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_camera_spec.resolution = camera_resolution
    semantic_camera_spec.postition = [0.0, 0.0, 0.0]
    semantic_camera_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_camera_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


# [/setup]

# This is wrapped such that it can be added to a unit test
def main(show_imgs=True, save_imgs=False):
    if save_imgs and not os.path.exists(output_path):
        os.mkdir(output_path)

    # [semantic id]
    scene_filepath = "/media/phong/Data2TB/dataset/Replica/Replica-Dataset/dataset/apartment_0/habitat/mesh_semantic.ply"
    # create the simulator and render flat shaded scene
    cfg = make_configuration(scene_file=scene_filepath)
    sim = habitat_sim.Simulator(cfg)

    #agent_transform = place_agent(sim)  # noqa: F841
    get_obs(sim, show_imgs, save_imgs)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-images", dest="show_images", action="store_false")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(show_images=True, save_images=True)
    args = parser.parse_args()
    main(show_imgs=args.show_images, save_imgs=args.save_images)
