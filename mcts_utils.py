from tkinter import N
from torchvision.transforms import functional as TF
import math
from multiprocessing import Pool
import time
from colorama import Fore
import numpy as np
import cv2
import torch
from environment import Environment

from pynvml import *

import utils
from dataset import LifelongEvalDataset, GraspDataset
from action_utils import adjust_push_start_point
from models import reinforcement_net, GraspNet
from vision.efficientnet import EfficientNet
from constants import (
    IMAGE_OBJ_GRASP_CENTER_SIZE,
    PIXEL_SIZE,
    GRIPPER_PUSH_ADD_PIXEL,
    PUSH_DISTANCE_PIXEL,
    TARGET_LOWER,
    TARGET_UPPER,
    IMAGE_PAD_WIDTH,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    NUM_ROTATION,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    BG_THRESHOLD,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
    WORKSPACE_PUSH_BORDER,
)


class MCTSHelper:
    """
    Simulate the state after push actions.
    Evaluation the grasp rewards.
    """

    def __init__(self, grasp_model_path, grasp_eval_model_path, seed: int):
        self.np_rng = np.random.default_rng(seed)
        self.pool = Pool(8)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Initialize Grasp Q Evaluation
        self.grasp_model = reinforcement_net()
        self.grasp_model.load_state_dict(torch.load(grasp_model_path)["model"], strict=False)
        self.grasp_model = self.grasp_model.to(self.device)
        self.grasp_model.eval()

        # Initialize Grasp Classification
        self.grasp_eval_model = EfficientNet.from_name("efficientnet-b0", in_channels=1, num_classes=1)
        self.grasp_eval_model.load_state_dict(torch.load(grasp_eval_model_path)["model"], strict=False)
        self.grasp_eval_model = self.grasp_eval_model.to(self.device)
        self.grasp_eval_model.eval()

        # tensor
        self.lower0 = torch.tensor(WORKSPACE_LIMITS[0][0] + WORKSPACE_PUSH_BORDER, device=self.device)
        self.upper0 = torch.tensor(WORKSPACE_LIMITS[0][1] - WORKSPACE_PUSH_BORDER, device=self.device)
        self.lower1 = torch.tensor(WORKSPACE_LIMITS[1][0] + WORKSPACE_PUSH_BORDER, device=self.device)
        self.upper1 = torch.tensor(WORKSPACE_LIMITS[1][1] - WORKSPACE_PUSH_BORDER, device=self.device)

        self.move_recorder = {}
        self.simulation_recorder = {}
        # self.predefine_grasp()

    def set_env(self, env: Environment):
        self.env = env
        self.main_id = torch.arange(0, 1, device=self.device)
        # use the second (anyone except the first) env to sample actions
        self.other_id = torch.arange(1, 2, device=self.device)
        self.all_other_id = torch.arange(1, env.num_envs, device=self.device)

    def reset(self):
        self.move_recorder = {}
        self.simulation_recorder = {}

    def simulate(self, env_ids, actions: list, object_states, restore_state=True):
        # print(f"simulate with {len(actions)} actions")
        assert len(env_ids) == len(actions)

        save_restore_sign = torch.ones(len(env_ids), dtype=torch.bool)

        if restore_state:
            success, failed_idx = self.env.restore_object_states(env_ids, object_states)
            if not success:
                save_restore_sign[failed_idx] = False
            if torch.any(save_restore_sign) == False:
                print(Fore.RED + f"all restore failed")
                return None, [False] * len(env_ids)

        pose0 = []
        pose1 = []
        for action in actions:
            primitive_position = [action[0][0], action[0][1], 0.01]
            primitive_position_end = [action[1][0], action[1][1], 0.01]
            pose0.append(primitive_position)
            pose1.append(primitive_position_end)
        pose0 = np.asarray(pose0)
        pose0[:, 0] = pose0[:, 0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]
        pose0[:, 1] = pose0[:, 1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
        pose1 = np.asarray(pose1)
        pose1[:, 0] = pose1[:, 0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]
        pose1[:, 1] = pose1[:, 1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]

        success_state = self.env.push_idx(
            env_ids[save_restore_sign], pose0[save_restore_sign.numpy()], pose1[save_restore_sign.numpy()]
        )
        save_restore_sign[save_restore_sign.clone()] &= success_state.cpu()
        success, all_states, non_static_idx = self.env.save_object_states(env_ids)
        if not success:
            save_restore_sign[non_static_idx] = False
        if torch.any(save_restore_sign) == False:
            print(Fore.RED + f"all push or save failed")
            return None, [False] * len(env_ids)

        # Check if all objects are still in workspace, exclude workspace and robot
        in_lower0 = torch.all(all_states[:, 2:, 0] > self.lower0, dim=1)
        in_upper0 = torch.all(all_states[:, 2:, 0] < self.upper0, dim=1)
        in_lower1 = torch.all(all_states[:, 2:, 1] > self.lower1, dim=1)
        in_upper1 = torch.all(all_states[:, 2:, 1] < self.upper1, dim=1)
        in_range = torch.logical_and(in_lower0, in_upper0)
        in_range = torch.logical_and(in_range, in_lower1)
        in_range = torch.logical_and(in_range, in_upper1)

        qualify = in_range.cpu()
        qualify = torch.logical_and(qualify, save_restore_sign)

        return all_states, qualify

    # @torch.no_grad()
    # def get_grasp_q(self, color_heightmap, depth_heightmap, post_checking=False, is_real=False):

    #     dataset = GraspDataset(color_heightmap, depth_heightmap, NUM_ROTATION)
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=NUM_ROTATION, shuffle=False, num_workers=8, drop_last=False
    #     )

    #     input_data = next(iter(data_loader))
    #     input_data = input_data.to(self.device)

    #     output_prob = self.grasp_model(input_data)
    #     grasp_predictions = output_prob.cpu().numpy()
    #     grasp_predictions = np.squeeze(grasp_predictions)

    #     # post process, only grasp one object, focus on blue object
    #     temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
    #     mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
    #     mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
    #     mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
    #     mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
    #     # focus on blue
    #     for rotate_idx in range(len(grasp_predictions)):
    #         grasp_predictions[rotate_idx][mask_pad != 255] = 0
    #     padding_width_start = IMAGE_PAD_WIDTH
    #     padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
    #     # only grasp one object
    #     kernel_big = np.ones((GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8)
    #     if is_real:  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
    #         threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
    #         threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
    #     else:
    #         threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
    #         threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
    #     for rotate_idx in range(len(grasp_predictions)):
    #         color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
    #         color_mask[color_mask == 0] = 1
    #         color_mask[color_mask == 255] = 0
    #         no_target_mask = color_mask
    #         bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
    #         no_target_mask[bg_mask == 255] = 0
    #         # only grasp one object
    #         invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
    #         # invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
    #         grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
    #             grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
    #         )
    #         grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

    #     # collision checking, only work for one level
    #     if post_checking:
    #         mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
    #         mask = 255 - mask
    #         mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
    #         check_kernel = np.ones((GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8)
    #         left_bound = math.floor((GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2)
    #         right_bound = math.ceil((GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2) + 1
    #         check_kernel[:, left_bound:right_bound] = 0
    #         for rotate_idx in range(len(grasp_predictions)):
    #             object_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
    #             invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
    #             invalid_mask[invalid_mask > 5] = 255
    #             # invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
    #             grasp_predictions[rotate_idx][invalid_mask > 128] = 0
    #     for rotate_idx in range(len(grasp_predictions)):
    #         grasp_predictions[rotate_idx] = utils.rotate(
    #             grasp_predictions[rotate_idx], -rotate_idx * (360.0 / NUM_ROTATION), False
    #         )
    #     grasp_predictions = grasp_predictions[
    #         :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
    #     ]

    #     best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    #     grasp_q_value = grasp_predictions[best_pix_ind]

    #     return grasp_q_value, best_pix_ind, grasp_predictions

    def predefine_grasp(self):
        """Copied from collect_grasp_data, assume target object is the first object"""
        grasps = []
        for x in range(0, IMAGE_OBJ_GRASP_CENTER_SIZE, 4):
            for y in range(0, IMAGE_OBJ_GRASP_CENTER_SIZE, 4):
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
                IMAGE_OBJ_GRASP_CENTER_SIZE / 2 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                IMAGE_OBJ_GRASP_CENTER_SIZE / 2 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            ]
        )
        self.grasps[:, 0:2] -= center
        print(f"predefined grasp {len(self.grasps)}")

    def propose_grasp(self):
        """Assume we want to grasp target in first env, and all other envs are free to use to simulate the grasping"""
        success, all_states, non_static_idx = self.env.save_object_states(self.main_id)
        assert success
        all_state = all_states[0]

        block_id = 2
        center = all_state[block_id, 0:2].clone().cpu().numpy()
        obj_height = 0.05
        grasps = self.grasps.copy()
        grasps[:, 0:2] += center
        all_other_obj_ids = list(range(all_state.shape[0]))
        all_other_obj_ids.remove(block_id)
        all_other_obj_ids = torch.tensor(all_other_obj_ids, device=self.env.device)
        ori_state = all_state[:, 0:2].clone()

        num_envs = len(self.all_other_id)
        num_group = len(grasps) // num_envs
        left_grasps = []
        for i in range(max(1, num_group)):
            current_grasps = grasps[i * num_envs : (i + 1) * num_envs]
            env_ids = torch.arange(1, len(current_grasps) + 1, device=self.device)
            restore_success, failed_idx = self.env.restore_object_states(env_ids, all_state)
            if len(failed_idx) > 0:
                valid = torch.ones(len(env_ids), dtype=torch.bool)
                valid[failed_idx] = False
                left_grasps.append(current_grasps[failed_idx.cpu()])
                current_grasps = current_grasps[valid]
                env_ids = env_ids[valid]

            current_success = self.env.grasp_idx(env_ids, current_grasps[:, 0:3], current_grasps[:, 3:])
            height_filter = self.env.all_state[env_ids, block_id, 2] > obj_height
            current_success = torch.logical_and(current_success, height_filter)
            height_filter = torch.all(self.env.all_state[env_ids][:, all_other_obj_ids, 2] < obj_height, dim=1)
            current_success = torch.logical_and(current_success, height_filter)

            if torch.any(current_success):
                success = current_success.nonzero().flatten()
                env_ids = env_ids[success]
                diff = self.env.all_state[env_ids, :, 0:2] - ori_state
                diff = diff.reshape(diff.shape[0], -1)
                i = torch.argmin(torch.sum(diff.abs(), dim=1)).item()
                grasp = current_grasps[success.cpu().numpy()][i]
                print(f"Propose a grasp {grasp}")
                return grasp

        # Leftover
        if len(left_grasps) > 0:
            if len(left_grasps) == 1:
                current_grasps = left_grasps[0]
            else:
                current_grasps = np.concatenate(left_grasps)
            assert len(current_grasps) <= num_envs
            env_ids = self.all_other_id.clone()
            restore_success, failed_idx = self.env.restore_object_states(env_ids, all_state)
            if len(failed_idx) > 0:
                valid = torch.ones(len(env_ids), dtype=torch.bool)
                valid[failed_idx] = False
                env_ids = env_ids[valid]
                assert len(env_ids) >= len(current_grasps)
            env_ids = env_ids[: len(current_grasps)]
            print("leftover", env_ids, current_grasps)
            current_success = self.env.grasp_idx(env_ids, current_grasps[:, 0:3], current_grasps[:, 3:])
            height_filter = self.env.all_state[env_ids, block_id, 2] > obj_height
            current_success = torch.logical_and(current_success, height_filter)
            height_filter = torch.all(self.env.all_state[env_ids][:, all_other_obj_ids, 2] < obj_height, dim=1)
            current_success = torch.logical_and(current_success, height_filter)

            if torch.any(current_success):
                success = current_success.nonzero().flatten()
                env_ids = env_ids[success]
                diff = self.env.all_state[env_ids, :, 0:2] - ori_state
                diff = diff.reshape(diff.shape[0], -1)
                i = torch.argmin(torch.sum(diff.abs(), dim=1)).item()
                grasp = current_grasps[success.cpu().numpy()][i]
                print(f"Propose a grasp {grasp}")
                return grasp

        print("Could not find a grasp")
        return None

    def get_grasp_q_batch(self, color_heightmaps, depth_heightmaps, post_checking=False, is_real=False):
        grasp_values = []

        for color, depth in zip(color_heightmaps, depth_heightmaps):
            q, _, _ = self.get_grasp_q(color, depth, post_checking=post_checking, is_real=is_real)
            grasp_values.append(q)

        return grasp_values

    @torch.no_grad()
    def get_grasp_q(self, color_heightmap, depth_heightmap, post_checking=False, is_real=False):
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        output_prob = self.grasp_model(input_color_data, input_depth_data, True, -1, False)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,]
                )
            else:
                grasp_predictions = np.concatenate(
                    (grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,],), axis=0,
                )

        # post process, only grasp one object, focus on blue object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask_pad != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones((GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8)
        if is_real:  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = GRIPPER_GRASP_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
        else:
            threshold_big = GRIPPER_GRASP_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = GRIPPER_GRASP_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
        for rotate_idx in range(len(grasp_predictions)):
            color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            color_mask[color_mask == 0] = 1
            color_mask[color_mask == 255] = 0
            no_target_mask = color_mask
            bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            no_target_mask[bg_mask == 255] = 0
            # only grasp one object
            invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
            invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, only work for one level
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            check_kernel = np.ones((GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8)
            left_bound = math.floor((GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2)
            right_bound = math.ceil((GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2) + 1
            check_kernel[:, left_bound:right_bound] = 0
            for rotate_idx in range(len(grasp_predictions)):
                object_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
                invalid_mask[invalid_mask > 5] = 255
                invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
                grasp_predictions[rotate_idx][invalid_mask > 128] = 0
        grasp_predictions = grasp_predictions[
            :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
        ]

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return grasp_q_value, best_pix_ind, grasp_predictions

    @torch.no_grad()
    def grasp_eval(self, depth):
        depth = np.copy(depth)

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth.shape = (depth.shape[0], depth.shape[1], 1)
        depth[:, :, 0] = (depth[:, :, 0] - image_mean[0]) / image_std[0]

        # Construct minibatch of size 1 (b,c,h,w)
        depth.shape = (
            1,
            1,
            depth.shape[0],
            depth.shape[1],
        )
        depth = torch.from_numpy(depth.astype(np.float32))
        depth = depth.to(self.device)

        # Pass input data through model
        output_prob = self.grasp_eval_model(depth)

        grasp_value = torch.sigmoid(output_prob)[0, 0].cpu().item()

        return grasp_value

    @torch.no_grad()
    def grasp_eval_batch(self, depths, batch_size=128):
        depth = np.stack(depths, axis=0)

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        # Construct minibatch (b,c,h,w)
        depth.shape = (depth.shape[0], 1, depth.shape[1], depth.shape[2])
        depth[:, 0, :, :] = (depth[:, 0, :, :] - image_mean[0]) / image_std[0]
        depth = torch.from_numpy(depth.astype(np.float32))
        num_batch = int(math.ceil(depth.size(0) / batch_size))
        results = []
        for i in range(num_batch):
            batch_depth = depth[i * batch_size : (i + 1) * batch_size]
            batch_depth = batch_depth.to(self.device)

            # Pass input data through model
            output_prob = self.grasp_eval_model(batch_depth)

            grasp_value = torch.sigmoid(output_prob)[:, 0].cpu().tolist()
            results.extend(grasp_value)

        return results

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, is_push=False):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations / 4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row * 4 + canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(
                        prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0, 0, 255), 2,
                    )
                prediction_vis = utils.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations))
                if rotate_idx == best_pix_ind[0]:
                    center = np.array([[[int(best_pix_ind[2]), int(best_pix_ind[1])]]])
                    M = cv2.getRotationMatrix2D(
                        (prediction_vis.shape[1] // 2, prediction_vis.shape[0] // 2,),
                        rotate_idx * (360.0 / num_rotations),
                        1,
                    )
                    center = cv2.transform(center, M)
                    center = np.transpose(center[0])
                    if is_push:
                        point_from = (int(center[0]), int(center[1]))
                        point_to = (int(center[0] + PUSH_DISTANCE_PIXEL), int(center[1]))
                        prediction_vis = cv2.arrowedLine(
                            prediction_vis, point_from, point_to, (100, 255, 0), 2, tipLength=0.2,
                        )
                    else:
                        prediction_vis = cv2.rectangle(
                            prediction_vis,
                            (
                                max(0, int(center[0]) - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2),
                                max(0, int(center[1]) - GRIPPER_GRASP_WIDTH_PIXEL // 2),
                            ),
                            (
                                min(prediction_vis.shape[1], int(center[0]) + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,),
                                min(prediction_vis.shape[0], int(center[1]) + GRIPPER_GRASP_WIDTH_PIXEL // 2,),
                            ),
                            (100, 255, 0),
                            1,
                        )
                        prediction_vis = cv2.rectangle(
                            prediction_vis,
                            (
                                max(0, int(center[0]) - GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2),
                                max(0, int(center[1]) - GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2),
                            ),
                            (
                                min(prediction_vis.shape[1], int(center[0]) + GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2,),
                                min(prediction_vis.shape[0], int(center[1]) + GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2,),
                            ),
                            (100, 100, 155),
                            1,
                        )
                background_image = utils.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations))
                prediction_vis = (
                    0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis
                ).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas

    def close_pool(self):
        self.pool.close()
        self.pool.join()


@torch.no_grad()
def from_maskrcnn(model, color_image, device, plot=False):
    """
    Use Mask R-CNN to do instance segmentation and output masks in binary format.
    Assume it works in real world
    """
    image = color_image.copy()
    image = TF.to_tensor(image)
    prediction = model([image.to(device)])[0]
    final_mask = np.zeros((720, 1280), dtype=np.uint8)
    labels = {}
    if plot:
        pred_mask = np.zeros((720, 1280), dtype=np.uint8)
    for idx, mask in enumerate(prediction["masks"]):
        # TODO, 0.9 can be tuned
        threshold = 0.7
        if prediction["scores"][idx] > threshold:
            # get mask
            img = mask[0].mul(255).byte().cpu().numpy()
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # too small
            if np.sum(img == 255) < 100:
                continue
            # overlap IoU 70%
            if np.sum(np.logical_and(final_mask > 0, img == 255)) > np.sum(img == 255) * 3 / 4:
                continue
            fill_pixels = np.logical_and(final_mask == 0, img == 255)
            final_mask[fill_pixels] = idx + 1
            labels[(idx + 1)] = prediction["labels"][idx].cpu().item()
            if plot:
                pred_mask[img > 0] = prediction["labels"][idx].cpu().item() * 10
                cv2.imwrite(str(idx) + "mask.png", img)
    if plot:
        cv2.imwrite("pred.png", pred_mask)
    print("Mask R-CNN: %d objects detected" % (len(np.unique(final_mask)) - 1), prediction["scores"].cpu())
    return final_mask, labels


if __name__ == "__main__":

    mcts_helper = MCTSHelper(
        None, "logs_grasp/snapshot-post-020000.reinforcement.pth", "logs_grasp/grasp_model-89.pth", 1234
    )

    depth_image1 = cv2.imread(
        "logs_mcts/mcts-2022-02-02-08-37-41-test10-test10-test10-test10-test10/data/depth-heightmaps/010001.0.depth.png",
        cv2.IMREAD_UNCHANGED,
    )
    depth_image1 = depth_image1 / 100000.0
    new_q_value = mcts_helper.grasp_eval(depth_image1)
    print(new_q_value)

    depth_image1 = cv2.imread(
        "logs_mcts/mcts-2022-02-02-08-37-41-test10-test10-test10-test10-test10/data/depth-heightmaps/001002.0.depth.png",
        cv2.IMREAD_UNCHANGED,
    )
    depth_image1 = depth_image1 / 100000.0
    new_q_value = mcts_helper.grasp_eval(depth_image1)
    print(new_q_value)
