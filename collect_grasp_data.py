import time
import datetime
import os
import glob
import numpy as np
from PIL import Image as im
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from constants import (
    IMAGE_OBJ_CROP_CENTER_SIZE,
    IMAGE_SIZE,
    NUM_ROTATION,
    PIXEL_SIZE,
    WORKSPACE_GRASP_BORDER,
)

import torch

from environment import Environment
from constants import (
    DEPTH_MIN,
    IMAGE_OBJ_CROP_SIZE,
    WORKSPACE_LIMITS,
)


class GraspDataCollector:
    def __init__(self, env, start_iter=0, end_iter=2000, base_directory=None, seed=0):
        # Objects have heights of 0.045 meters, so center should be less than 0.03
        self.height_upper = 0.03
        self.depth_min = DEPTH_MIN

        self.rng = np.random.default_rng(seed)

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        if base_directory is None:
            self.base_directory = os.path.join(
                os.path.abspath("logs_grasp"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
            )
        else:
            self.base_directory = base_directory
        print("Creating data logging session: %s" % (self.base_directory))
        self.overview_directory = os.path.join(self.base_directory, "overviews")
        self.color_directory = os.path.join(self.base_directory, "colors")
        self.depth_directory = os.path.join(self.base_directory, "depths")
        self.segm_directory = os.path.join(self.base_directory, "segms")
        self.action_directory = os.path.join(self.base_directory, "actions")
        self.label_directory = os.path.join(self.base_directory, "labels")

        if not os.path.exists(self.overview_directory):
            os.makedirs(self.overview_directory)
        if not os.path.exists(self.color_directory):
            os.makedirs(self.color_directory)
        if not os.path.exists(self.depth_directory):
            os.makedirs(self.depth_directory)
        if not os.path.exists(self.segm_directory):
            os.makedirs(self.segm_directory)
        if not os.path.exists(self.action_directory):
            os.makedirs(self.action_directory)
        if not os.path.exists(self.label_directory):
            os.makedirs(self.label_directory)

        self.iter = start_iter
        self.end_iter = end_iter

        self.main_id = torch.arange(1, device=env.device)
        self.grasp_ids = torch.arange(1, env.num_envs, device=env.device)
        self.env = env
        self.labels = []
        self.predefine_grasp()

    def reset_np_random(self, seed):
        self.rng = np.random.default_rng(seed)

    def save_overviews(self, iteration, image):
        color_image = im.fromarray(image, mode="RGB")
        color_image.save(f"{self.overview_directory}/{iteration:07}.over.png")

    def save_images_single_obj(
        self, iteration, color_image, depth_image, segm_image,
    ):
        color_image = im.fromarray(color_image, mode="RGB")
        color_image.save(f"{self.color_directory}/{iteration:07}.color.png")

        depth_image = np.round(depth_image * 100000).astype(np.uint16)
        depth_image = im.fromarray(depth_image, mode="I;16")
        depth_image.save(f"{self.depth_directory}/{iteration:07}.depth.png")

        segm_image = im.fromarray(segm_image, mode="L")
        segm_image.save(f"{self.segm_directory}/{iteration:07}.segm.png")

    def save_action(self, iteration, actions):
        np.savetxt(os.path.join(self.action_directory, "%07d.actions.txt" % (iteration)), actions, fmt="%s")

    def save_label(self, labels):
        np.savetxt(os.path.join(self.label_directory, "labels.txt"), labels, fmt="%s")

    def add_object_grasp_from_file(self, file_name):
        """Read position of objects and put to the workspace"""
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            block_files = []
            block_poses = []
            block_colors = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                block_pose = gymapi.Transform()
                block_pose.p = gymapi.Vec3(
                    float(file_content_curr_object[4]),
                    float(file_content_curr_object[5]),
                    float(file_content_curr_object[6]),
                )
                block_pose.r = gymapi.Quat.from_euler_zyx(
                    float(file_content_curr_object[7]),
                    float(file_content_curr_object[8]),
                    float(file_content_curr_object[9]),
                )
                block_color = gymapi.Vec3(
                    float(file_content_curr_object[1]),
                    float(file_content_curr_object[2]),
                    float(file_content_curr_object[3]),
                )
                block_files.append(file_content_curr_object[0])
                block_poses.append(block_pose)
                block_colors.append(block_color)

        block_type_ids = {
            "concave.urdf": list(range(2, 12)),
            "half-cube.urdf": list(range(12, 22)),
            "triangle.urdf": list(range(22, 32)),
            "cylinder.urdf": list(range(32, 42)),
            "half-cylinder.urdf": list(range(42, 52)),
            "rect.urdf": list(range(52, 62)),
            "cube.urdf": list(range(62, 72)),
        }

        obj_mesh_ids = []
        for idx, block_name in enumerate(block_files):
            obj_id = self.rng.choice(block_type_ids[block_name], 1)[0]
            block_type_ids[block_name].remove(obj_id)
            self.env.all_state[self.main_id, obj_id, 0] = block_poses[idx].p.x
            self.env.all_state[self.main_id, obj_id, 1] = block_poses[idx].p.y
            self.env.all_state[self.main_id, obj_id, 2] = block_poses[idx].p.z
            self.env.all_state[self.main_id, obj_id, 3] = block_poses[idx].r.x
            self.env.all_state[self.main_id, obj_id, 4] = block_poses[idx].r.y
            self.env.all_state[self.main_id, obj_id, 5] = block_poses[idx].r.z
            self.env.all_state[self.main_id, obj_id, 6] = block_poses[idx].r.w
            obj_mesh_ids.append(obj_id)

        indices = torch.tensor(obj_mesh_ids, dtype=torch.int32, device=self.env.device)
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim, self.env.actor_root_state_tensor, gymtorch.unwrap_tensor(indices), len(obj_mesh_ids),
        )
        success = len(self.env.wait_static(self.main_id)) == 0

        return obj_mesh_ids, success

    def add_object_grasp(self, env_id):
        """Randomly dropped objects to the workspace"""
        drop_height = 0.1
        obj_num = self.rng.choice([5, 6, 7, 8, 9, 10], p=[0.04, 0.06, 0.15, 0.2, 0.25, 0.3])
        obj_mesh_ids = self.rng.choice(self.env.block_idxs[0], obj_num, replace=False)

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        object_positions = []
        object_orientations = []
        success = True
        for object_idx in range(obj_num):
            body_id = obj_mesh_ids[object_idx]
            obj_sim_idx = torch.arange(body_id, body_id + 1, dtype=torch.int32, device=self.env.device)
            drop_x = 0.45 + self.rng.random() * 0.1
            drop_y = -0.05 + self.rng.random() * 0.1
            object_position = [drop_x, drop_y, drop_height]
            object_orientation = [
                0,
                0,
                2 * np.pi * self.rng.random(),
            ]
            adjust_angle = 2 * np.pi * self.rng.random()
            quat = gymapi.Quat.from_euler_zyx(
                float(object_orientation[0]), float(object_orientation[1]), float(object_orientation[2]),
            )
            self.env.all_state[env_id, body_id, 0] = object_position[0]
            self.env.all_state[env_id, body_id, 1] = object_position[1]
            self.env.all_state[env_id, body_id, 2] = object_position[2]
            self.env.all_state[env_id, body_id, 3] = quat.x
            self.env.all_state[env_id, body_id, 4] = quat.y
            self.env.all_state[env_id, body_id, 5] = quat.z
            self.env.all_state[env_id, body_id, 6] = quat.w
            self.env.gym.set_actor_root_state_tensor_indexed(
                self.env.sim, self.env.actor_root_state_tensor, gymtorch.unwrap_tensor(obj_sim_idx), 1,
            )
            success &= len(self.env.wait_static(env_id)) == 0

            count = 0
            while success:
                object_position = self.env.all_state[0, body_id, 0:3]
                if count > 20:
                    break
                # if overlap
                if object_position[2] > self.height_upper:
                    drop_x = np.cos(adjust_angle) * 0.01 + drop_x  # 2 cm
                    drop_y = np.sin(adjust_angle) * 0.01 + drop_y
                    object_position = [drop_x, drop_y, drop_height]
                    self.env.all_state[env_id, body_id, 0] = object_position[0]
                    self.env.all_state[env_id, body_id, 1] = object_position[1]
                    self.env.all_state[env_id, body_id, 2] = object_position[2]
                    self.env.all_state[env_id, body_id, 3] = quat.x
                    self.env.all_state[env_id, body_id, 4] = quat.y
                    self.env.all_state[env_id, body_id, 5] = quat.z
                    self.env.all_state[env_id, body_id, 6] = quat.w
                    self.env.gym.set_actor_root_state_tensor_indexed(
                        self.env.sim, self.env.actor_root_state_tensor, gymtorch.unwrap_tensor(obj_sim_idx), 1,
                    )
                else:
                    break
                count += 1
                success &= len(self.env.wait_static(env_id)) == 0
            if count > 20:
                object_position = [drop_x, drop_y, self.height_upper]
                self.env.all_state[env_id, body_id, 0] = object_position[0]
                self.env.all_state[env_id, body_id, 1] = object_position[1]
                self.env.all_state[env_id, body_id, 2] = object_position[2]
                self.env.all_state[env_id, body_id, 3] = quat.x
                self.env.all_state[env_id, body_id, 4] = quat.y
                self.env.all_state[env_id, body_id, 5] = quat.z
                self.env.all_state[env_id, body_id, 6] = quat.w
                self.env.gym.set_actor_root_state_tensor_indexed(
                    self.env.sim, self.env.actor_root_state_tensor, gymtorch.unwrap_tensor(obj_sim_idx), 1,
                )
                success &= len(self.env.wait_static(env_id)) == 0

            object_positions.append(object_position)
            object_orientations.append(quat)

        for object_idx in range(obj_num):
            body_id = obj_mesh_ids[object_idx]
            obj_sim_idx = torch.arange(body_id, body_id + 1, dtype=torch.int32, device=self.env.device)
            object_position = object_positions[object_idx]
            quat = object_orientations[object_idx]
            self.env.all_state[env_id, body_id, 0] = object_position[0]
            self.env.all_state[env_id, body_id, 1] = object_position[1]
            self.env.all_state[env_id, body_id, 2] = object_position[2]
            self.env.all_state[env_id, body_id, 3] = quat.x
            self.env.all_state[env_id, body_id, 4] = quat.y
            self.env.all_state[env_id, body_id, 5] = quat.z
            self.env.all_state[env_id, body_id, 6] = quat.w
            self.env.gym.set_actor_root_state_tensor_indexed(
                self.env.sim, self.env.actor_root_state_tensor, gymtorch.unwrap_tensor(obj_sim_idx), 1,
            )
            success &= len(self.env.wait_static(env_id)) == 0

        return obj_mesh_ids, success

    def is_valid(self, body_ids):
        """Decide randomly dropped objects in the valid state."""
        object_positions = self.env.all_state[0, body_ids, 0:3]

        # Check height
        if torch.any(object_positions[..., 2] > self.height_upper):
            print(f"Height is wrong. Skip! {object_positions[..., 2]} > {self.height_upper}")
            return False

        # Check range
        if (
            torch.any(object_positions[..., 0] < WORKSPACE_LIMITS[0][0] + WORKSPACE_GRASP_BORDER)
            or torch.any(object_positions[..., 0] > WORKSPACE_LIMITS[0][1] - WORKSPACE_GRASP_BORDER)
            or torch.any(object_positions[..., 1] < WORKSPACE_LIMITS[1][0] + WORKSPACE_GRASP_BORDER)
            or torch.any(object_positions[..., 1] > WORKSPACE_LIMITS[1][1] - WORKSPACE_GRASP_BORDER)
        ):
            print(f"Out of bounds. Skip! {object_positions[..., 0:2]}")
            return False

        # Check orientation
        object_orientations = self.env.all_state[0, body_ids, 3:7]
        for i in range(len(body_ids)):
            r = gymapi.Quat(
                object_orientations[i][0],
                object_orientations[i][1],
                object_orientations[i][2],
                object_orientations[i][3],
            )
            r = r.to_euler_zyx()
            if abs(r[0]) > 1e-2 or abs(r[1]) > 1e-2:
                print(f"Wrong orientation. Skip! {r}")
                return False

        return True

    def grasp_and_record(self, block_id, object_states, color, depth, segm):
        center = object_states[self.main_id, block_id, 0:2].clone().cpu().numpy()[0]
        obj_height = 0.05
        grasps = self.grasps.copy()
        grasps[:, 0:2] += center
        success = np.zeros(len(grasps))
        all_other_ids = list(range(object_states.shape[1]))
        all_other_ids.remove(block_id)
        all_other_ids = torch.tensor(all_other_ids, device=self.env.device)

        num_sets = len(self.grasp_ids)
        assert (
            len(grasps) // num_sets == 1
        )  # TODO: assume we can process all grasps with exactly same number as free envs
        for i in range(len(grasps) // num_sets):
            print(object_states.shape)
            restore_success, _ = self.env.restore_object_states(self.grasp_ids, object_states)
            if not restore_success:
                return False
            current_grasps = grasps[i * num_sets : (i + 1) * num_sets]
            current_success = self.env.grasp_idx(self.grasp_ids, current_grasps[:, 0:3], current_grasps[:, 3:])
            height_filter = self.env.all_state[self.grasp_ids, block_id, 2] > obj_height
            current_success = torch.logical_and(current_success, height_filter)
            height_filter = torch.all(self.env.all_state[self.grasp_ids][:, all_other_ids, 2] < obj_height, dim=1)
            current_success = torch.logical_and(current_success, height_filter)
            success[i * num_sets : (i + 1) * num_sets] = current_success.cpu().int()

        # images
        x = int(round((center[0] - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE))
        y = int(round((center[1] - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE))
        color_image = color[
            max(0, x - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, x + IMAGE_OBJ_CROP_SIZE // 2),
            max(0, y - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, y + IMAGE_OBJ_CROP_SIZE // 2),
            :,
        ]
        depth_image = depth[
            max(0, x - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, x + IMAGE_OBJ_CROP_SIZE // 2),
            max(0, y - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, y + IMAGE_OBJ_CROP_SIZE // 2),
        ]
        segm_image = segm[
            max(0, x - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, x + IMAGE_OBJ_CROP_SIZE // 2),
            max(0, y - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, y + IMAGE_OBJ_CROP_SIZE // 2),
        ]
        if color_image.shape[0] < IMAGE_OBJ_CROP_SIZE or color_image.shape[1] < IMAGE_OBJ_CROP_SIZE:
            pad = [[0, 0], [0, 0], [0, 0]]
            if x - IMAGE_OBJ_CROP_SIZE // 2 < 0:
                pad[0][0] = IMAGE_OBJ_CROP_SIZE // 2 - x
            if y - IMAGE_OBJ_CROP_SIZE // 2 < 0:
                pad[1][0] = IMAGE_OBJ_CROP_SIZE // 2 - y
            if x + IMAGE_OBJ_CROP_SIZE // 2 > IMAGE_SIZE:
                pad[0][1] = (x + IMAGE_OBJ_CROP_SIZE // 2) - IMAGE_SIZE
            if y + IMAGE_OBJ_CROP_SIZE // 2 > IMAGE_SIZE:
                pad[1][1] = (y + IMAGE_OBJ_CROP_SIZE // 2) - IMAGE_SIZE
            color_image = np.pad(color_image, pad, "constant", constant_values=0)
            depth_image = np.pad(depth_image, pad[:2], "constant", constant_values=0)
            segm_image = np.pad(segm_image, pad[:2], "constant", constant_values=0)
        self.save_images_single_obj(self.iter, color_image, depth_image, segm_image)

        # actions
        success = np.expand_dims(success, axis=1)
        actions = np.concatenate((success, grasps), axis=1)
        self.save_action(self.iter, actions)

        # labels
        self.labels.append([np.any(success) * 1])
        self.save_label(self.labels)

        return True

    def predefine_grasp(self):
        grasps = []
        for x in range(0, IMAGE_OBJ_CROP_CENTER_SIZE, 4):
            for y in range(0, IMAGE_OBJ_CROP_CENTER_SIZE, 4):
                for yaw in range(NUM_ROTATION // 2):
                    grasps.append(
                        [
                            x * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                            y * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                            0.01,
                            yaw * (np.pi / (NUM_ROTATION // 2)),
                        ]
                    )

        self.grasps = np.asarray(grasps)
        center = np.asarray(
            [
                IMAGE_OBJ_CROP_CENTER_SIZE / 2 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                IMAGE_OBJ_CROP_CENTER_SIZE / 2 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            ]
        )
        self.grasps[:, 0:2] -= center

        print(f"predefined grasp {len(self.grasps)}")


def set_args():
    custom_parameters = [
        {
            "name": "--controller",
            "type": str,
            "default": "ik",
            "help": "Controller to use for Franka. Options are {ik, osc}",
        },
        {"name": "--num_envs", "type": int, "default": 1153, "help": "Number of environments to create"},
        {
            "name": "--test_case",
            "type": str,
            "default": "test-cases/grasp/collect_grasp.txt",
            "help": "Test case to create",
        },
    ]
    args = gymutil.parse_arguments(description="UR5e Grasp", custom_parameters=custom_parameters, headless=True)

    return args


if __name__ == "__main__":

    # TODO: this can be improved maybe, the input should not be just a single depth,
    # even the target is always in the center, we should somehow also tell network that we focus on the target.
    # for example, all target object can have a unique depth info comparing to others, like += 0.1

    torch.manual_seed(1234)

    torch.set_printoptions(precision=4, sci_mode=False)

    args = set_args()
    env = Environment(args)
    reset_ids = torch.arange(env.num_envs, device=env.device)

    seed = 0
    collector = GraspDataCollector(env, start_iter=0, end_iter=2000)

    # ===== generate random cases =====
    # while collector.iter < collector.end_iter:
    #     print(f"-----Generating: {collector.iter + 1}/{collector.end_iter}-----")
    #     collector.reset_np_random(seed)
    #     env.reset_idx(reset_ids)

    #     obj_mesh_ids, success = collector.add_object_grasp(collector.main_id)
    #     if success and collector.is_valid(obj_mesh_ids):
    #         obj_states = env.save_object_states(collector.main_id)
    #         torch.save(
    #             {"obj_ids": obj_mesh_ids, "obj_states": obj_states}, f"logs_grasp/random{collector.iter}.pt"
    #         )
    #         collector.iter += 1
    #     seed += 1
    # ===== generate random cases =====

    # ===== generate hard cases =====
    # case_idx = 2240
    # cases = sorted(glob.glob("test-cases/random-hard/*.txt"))
    # while collector.iter < collector.end_iter:
    #     print(f"-----Generating: {collector.iter + 1}/{collector.end_iter}-----")
    #     collector.reset_np_random(seed)
    #     env.reset_idx(reset_ids)

    #     obj_mesh_ids, success = collector.add_object_grasp_from_file(cases[case_idx])
    #     print(obj_mesh_ids)
    #     if success and collector.is_valid(obj_mesh_ids):
    #         obj_states = env.save_object_states(collector.main_id)
    #         torch.save({"obj_ids": obj_mesh_ids, "obj_states": obj_states}, f"hard{collector.iter}.pt")
    #         collector.iter += 1
    #     case_idx += 1
    #     seed += 1
    # ===== generate random cases =====

    # ===== collect grasp from file =====
    case_idx = 0
    cases = sorted(glob.glob("test-cases/grasp/test/hard/*.pt"))
    while case_idx < len(cases):

        print(f"-----Collecting: {collector.iter + 1} (case {case_idx})-----")
        collector.reset_np_random(seed)
        env.reset_idx(reset_ids)

        info = torch.load(cases[case_idx])
        obj_mesh_ids = info["obj_ids"]
        obj_states = info["obj_states"].cuda()
        success, _ = env.restore_object_states(collector.main_id, obj_states)

        if success:
            images = env.render_camera(collector.main_id, True, True, True)
            # color_images, depth_images, segm_images = env.render_camera(collector.main_id, True, True, True)
            color, depth, segm = images[0][0], images[1][0], images[2][0]
            collector.save_overviews(collector.iter, color)

            for idx, obj_id in enumerate(obj_mesh_ids):
                # if idx != 1:
                #     continue
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                print(f"{idx}/{len(obj_mesh_ids)} {current_time}")
                # pose = [[0.5027192831039429, -0.05645139998197557, 0.01]] * len(collector.grasp_ids)
                # angle = [[0.7853981633974483]] * len(collector.grasp_ids)
                # env.restore_object_states(collector.grasp_ids, obj_states)
                # sucess = env.grasp_idx(collector.grasp_ids, pose, angle)
                # print(env.all_state[collector.grasp_ids, obj_id, :3])
                # print(sucess.cpu().int())
                success = collector.grasp_and_record(obj_id, obj_states, color, depth, segm)
                if success:
                    collector.iter += 1
        case_idx += 1
        seed += 1
    # ===== collect grasp from file =====
