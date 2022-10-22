import math
from multiprocessing import Pool
import sys
import time

import numpy as np
from PIL import Image as im
from colorama import Fore
from constants import IS_REAL


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from action_utils import (
    load_pre_defined_actions,
    quaternion_to_euler_angle_vectorized,
    sample_actions,
    sample_actions_parallel,
    sample_pre_defined_actions,
    sample_pre_defined_actions_parallel,
)
from constants import (
    IMAGE_OBJ_CROP_SIZE,
    IMAGE_SIZE,
    PIXEL_SIZE,
    WORKSPACE_LIMITS,
    WORKSPACE_PUSH_BORDER,
)

import torch


from pynvml import *

nvmlInit()


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class Environment:
    def __init__(self, args) -> None:
        # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()
        self.device = args.sim_device if args.use_gpu_pipeline else "cpu"
        self.physics_engine = args.physics_engine
        self.test_case = args.test_case
        self.aggregate_mode = True

        self._define_constant_params()

        # configure sim
        self.controller = args.controller
        sim_params = self._set_sim_params(args)
        self.dt = sim_params.dt

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create env
        self.num_envs = args.num_envs
        self._create_envs(self.num_envs, 1.0)
        self.gym.prepare_sim(self.sim)

        # create viewer
        self.viewer = None
        if not args.headless:
            self._create_viewer()

        # get gym GPU state tensors
        self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5e")
        _net_force = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors
        vec_actor_root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor).view(self.num_envs, -1, 13)
        vec_dof_state = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, -1, 2)
        vec_rigid_body_state_tensor = gymtorch.wrap_tensor(self.rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # vec_jacobian = gymtorch.wrap_tensor(_jacobian)
        vec_net_force = gymtorch.wrap_tensor(_net_force).view(self.num_envs, -1, 3)

        # create ur5e variables   0, -1.57, -1.57, -1.57, 1.57, 1.57
        self.ur5e_home_dof_pos = to_torch(
            [
                0,
                -1.57,
                -1.57,
                -1.57,
                1.57,
                1.57,
                self.gripper_angle_close,
                -self.gripper_angle_close,
                self.gripper_angle_close,
                self.gripper_angle_close,
                self.gripper_angle_close,
                -self.gripper_angle_close,
            ],
            device=self.device,
        ).unsqueeze(0)
        self.ur5e_dof_targets = torch.zeros((self.num_envs, self.num_ur5e_dofs), dtype=torch.float, device=self.device)
        self.ur5e_effort_action = torch.zeros_like(self.ur5e_dof_targets)
        self.ur5e_dof_state = vec_dof_state
        self.ur5e_dof_pos = self.ur5e_dof_state[..., 0]
        self.ur5e_dof_vel = self.ur5e_dof_state[..., 1]
        self.gripper_main_joint_pos = self.ur5e_dof_state[:, self.gripper_main_joint_handle, 0].view(self.num_envs, -1)
        self.gripper_main_joint_vel = self.ur5e_dof_state[:, self.gripper_main_joint_handle, 1].view(self.num_envs, -1)
        self.gripper_center_pos = vec_rigid_body_state_tensor[:, self.gripper_center_handle, 0:3]
        self.gripper_center_rot = vec_rigid_body_state_tensor[:, self.gripper_center_handle, 3:7]
        self.gripper_center_vel = vec_rigid_body_state_tensor[:, self.gripper_center_handle, 7:]
        # self.j_eef = vec_jacobian[:, self.ur5e_ee_center_link_index - 1, :, :6]  # first 6 joints
        self.force_l_z = vec_net_force[
            :, self.gripper_handles[0], 2,
        ]
        self.force_r_z = vec_net_force[
            :, self.gripper_handles[1], 2,
        ]
        self.force_l_y = vec_net_force[
            :, self.gripper_handles[0], 1,
        ]
        self.force_r_y = vec_net_force[
            :, self.gripper_handles[1], 1,
        ]

        # create block variables
        self.block_state = vec_actor_root_state_tensor[:, 2:]
        self.all_state = vec_actor_root_state_tensor

        # record indices
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.global_ur5e_indices = self.global_indices[:, 1].flatten()

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f"total    : {info.total / (1024 ** 2)}")
        print(f"free     : {info.free / (1024 ** 2)}")
        print(f"used     : {info.used / (1024 ** 2)}")

    def _create_envs(self, num_envs, spacing):

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        # plane_params.static_friction = 1.5
        # plane_params.dynamic_friction = 0.9
        # plane_params.restitution = 0.5
        self.gym.add_ground(self.sim, plane_params)

        # create workspace
        dim = WORKSPACE_LIMITS[0][1] - WORKSPACE_LIMITS[0][0]
        workspace_dims = gymapi.Vec3(dim, dim, 0.001)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.linear_damping = 0.5
        asset_options.angular_damping = 0.5
        workspace_asset = self.gym.create_box(
            self.sim, workspace_dims.x, workspace_dims.y, workspace_dims.z, asset_options
        )
        workspace_props = self.gym.get_asset_rigid_shape_properties(workspace_asset)
        if IS_REAL:
            workspace_props[0].friction = 1.1
        else:
            workspace_props[0].friction = 1.1
        workspace_props[0].restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(workspace_asset, workspace_props)
        workspace_pose = gymapi.Transform()
        workspace_pose.p = gymapi.Vec3(0.5, 0.0, 0.0005)

        # load UR5e asset
        # asset_root = "../../assets"
        # ur5e_asset_file = "urdf/ur5e/ur5e_gripper.urdf"
        asset_root = "assets"
        ur5e_asset_file = "ur5e/ur5e_gripper.urdf"
        # ur5e_push_asset_file = "urdf/ur5e/ur5e_stick.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = True
        ur5e_asset = self.gym.load_asset(self.sim, asset_root, ur5e_asset_file, asset_options)
        ur5e_start_pose = gymapi.Transform()
        ur5e_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur5e_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        ur5e_props = self.gym.get_asset_rigid_shape_properties(ur5e_asset)
        for prop in ur5e_props:
            prop.friction = 0.6
        self.gym.set_asset_rigid_shape_properties(ur5e_asset, ur5e_props)
        # ur5e_props[0].friction = 0.9
        # ur5e_push_asset = self.gym.load_asset(self.sim, asset_root, ur5e_push_asset_file, asset_options)
        # ur5e_push_start_pose = gymapi.Transform()
        # ur5e_push_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        # ur5e_push_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # configure ur5e dofs
        self.num_ur5e_dofs = self.gym.get_asset_dof_count(ur5e_asset)
        ur5e_dof_props = self.gym.get_asset_dof_properties(ur5e_asset)
        # use position drive for all dofs
        ur5e_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        ur5e_dof_props["stiffness"][:6].fill(5000.0)
        ur5e_dof_props["damping"][:6].fill(200.0)
        # grippers
        ur5e_dof_props["stiffness"][6:].fill(1e6)
        ur5e_dof_props["damping"][6:].fill(80)

        # # number of links and dof are made to be the same
        # ur5e_push_dof_props = self.gym.get_asset_dof_properties(ur5e_push_asset)
        # # use position drive for all dofs
        # ur5e_push_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        # ur5e_push_dof_props["stiffness"][:6].fill(5000.0)
        # ur5e_push_dof_props["damping"][:6].fill(200.0)

        # get link index of center of gripper, which we will use as end effector
        # self.ur5e_ee_center_link_index = self.gym.find_asset_rigid_body_index(ur5e_asset, "dummy_center_indicator_link")

        # configure env grid
        num_per_row = int(math.sqrt(num_envs))
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % num_envs)

        # create block assets
        block_assets, block_poses, block_colors = self.add_object_from_file(self.test_case)
        num_blocks = len(block_assets)

        # compute aggregate size
        num_ur5e_bodies = self.gym.get_asset_rigid_body_count(ur5e_asset)
        num_ur5e_shapes = self.gym.get_asset_rigid_shape_count(ur5e_asset)
        num_block_bodies = 0
        num_block_shapes = 0
        for asset in block_assets:
            num_block_bodies += self.gym.get_asset_rigid_body_count(asset)
            num_block_shapes += self.gym.get_asset_rigid_shape_count(asset)
        max_agg_bodies = num_ur5e_bodies + num_block_bodies + 1
        max_agg_shapes = num_ur5e_shapes + num_block_shapes + 1
        # num_ur5e_push_bodies = self.gym.get_asset_rigid_body_count(ur5e_push_asset)
        # num_ur5e_push_shapes = self.gym.get_asset_rigid_shape_count(ur5e_push_asset)
        # max_agg_push_bodies = num_ur5e_push_bodies + num_block_bodies + 1
        # max_agg_push_shapes = num_ur5e_push_shapes + num_block_shapes + 1

        # configure camera # TODO: enable tensor if actions can be sampled on GPU
        camera_props = gymapi.CameraProperties()
        camera_props.width = 224
        camera_props.height = 224
        camera_props.horizontal_fov = 0.02578  # TODO: find source to compute this
        camera_props.near_plane = 999.7
        camera_props.far_plane = 1001.0
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(0.5, 0, 999.8)
        camera_transform.r = gymapi.Quat.from_euler_zyx(np.radians(180.0), np.radians(90.0), 0)

        # change lights
        self.gym.set_light_parameters(
            self.sim, 0, gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0, 0, 0)
        )
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))

        self.ur5es = []
        self.envs = []
        self.cameras = []
        self.gripper_center_idxs = []
        self.block_idxs = []
        self.default_block_state = []

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # add camera
            cam = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_transform(cam, env_ptr, camera_transform)

            if self.aggregate_mode:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
                # self.gym.begin_aggregate(env_ptr, max_agg_push_bodies, max_agg_push_shapes, True)

            # add workspace
            workspace_handle = self.gym.create_actor(env_ptr, workspace_asset, workspace_pose, "workspace", i, 0, 0)
            self.gym.set_rigid_body_color(
                env_ptr, workspace_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0)
            )

            # add robot
            ur5e_actor = self.gym.create_actor(env_ptr, ur5e_asset, ur5e_start_pose, "ur5e", i, 1, 1)
            # ur5e_actor = self.gym.create_actor(env_ptr, ur5e_push_asset, ur5e_push_start_pose, "ur5e", i, 1, 1)
            self.gym.set_actor_dof_properties(env_ptr, ur5e_actor, ur5e_dof_props)

            # add blocks
            local_block_idxs = []
            for ib in range(num_blocks):
                if ib == 0:
                    segm_id = 255
                else:
                    segm_id = 50 + 10 * (ib - 1)

                pose = block_poses[ib]
                block_handle = self.gym.create_actor(env_ptr, block_assets[ib], pose, f"block{segm_id}", i, 0, segm_id)
                self.gym.set_rigid_body_color(
                    env_ptr, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, block_colors[ib]
                )

                # get local index of block
                block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_ENV)
                local_block_idxs.append(block_idx)
                self.default_block_state.append(
                    [pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w, 0, 0, 0, 0, 0, 0]
                )

            if self.aggregate_mode:
                self.gym.end_aggregate(env_ptr)

            self.block_idxs.append(local_block_idxs)
            self.envs.append(env_ptr)
            self.ur5es.append(ur5e_actor)
            self.cameras.append(cam)
            gripper_center_idx = self.gym.find_actor_rigid_body_index(
                env_ptr, ur5e_actor, "dummy_center_indicator_link", gymapi.DOMAIN_SIM
            )
            self.gripper_center_idxs.append(gripper_center_idx)

        self.num_ur5es = len(self.ur5es)
        self.num_blocks = num_blocks
        self.num_actors = 1 + 1 + num_blocks  # per env

        self.gripper_center_handle = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.ur5es[0], "dummy_center_indicator_link", gymapi.DOMAIN_ENV
        )
        self.gripper_main_joint_handle = self.gym.find_actor_dof_index(
            self.envs[0], self.ur5es[0], "finger_joint", gymapi.DOMAIN_ENV
        )

        link_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.ur5es[0])
        finger_names = [name for name in link_names if "pad" in name]
        self.gripper_handles = [
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5es[0], name) for name in finger_names
        ]

        self.target_handle = self.gym.find_actor_index(self.envs[0], "block255", gymapi.DOMAIN_ENV)

        self.default_block_state = to_torch(self.default_block_state, device=self.device, dtype=torch.float).view(
            self.num_envs, self.num_blocks, 13
        )

    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")

        # point camera at target env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        # target_env = self.envs[self.num_envs // 2 + int(math.sqrt(self.num_envs)) // 2]  # target
        target_env = self.envs[0]  # first
        self.gym.viewer_camera_look_at(self.viewer, target_env, cam_pos, cam_target)

        # lower0 = WORKSPACE_LIMITS[0][0] + WORKSPACE_PUSH_BORDER
        # upper0 = WORKSPACE_LIMITS[0][1] - WORKSPACE_PUSH_BORDER
        # lower1 = WORKSPACE_LIMITS[1][0] + WORKSPACE_PUSH_BORDER
        # upper1 = WORKSPACE_LIMITS[1][1] - WORKSPACE_PUSH_BORDER
        # vertices = [
        #     [lower0, lower1, 0.02, lower0, upper1, 0.02],
        #     [lower0, lower1, 0.02, upper0, lower1, 0.02],
        #     [upper0, upper1, 0.02, lower0, upper1, 0.02],
        #     [upper0, upper1, 0.02, upper0, lower1, 0.02],
        # ]
        # colors = [[0.85, 0.1, 0.1], [0.85, 0.1, 0.1], [0.85, 0.1, 0.1], [0.85, 0.1, 0.1]]
        # for env_ptr in self.envs:
        #     self.gym.add_lines(self.viewer, env_ptr, len(vertices), vertices, colors)

    def _define_constant_params(self):
        # DLS IK params
        self.damping = 0.05

        # Simple IK params, from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
        self.ik_d0 = 0.1625
        self.ik_a1 = -0.425
        self.ik_a2 = -0.3922
        self.ik_d3 = 0.1333
        self.ik_d4 = 0.0997
        self.ik_d5 = 0.0996
        self.ik_offset = 0.1725
        self.ik_a1_2 = self.ik_a1 ** 2
        self.ik_a2_2 = self.ik_a2 ** 2
        self.ik_d3_2 = self.ik_d3 ** 2
        self.ik_a1_a2 = self.ik_a1 * self.ik_a2

        # gripper variables
        self.gripper_center_offset = 0.02
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_angle_signs = to_torch([1, -1, 1, 1, 1, -1], device=self.device,).unsqueeze(0)
        self.gripper_angle_open_tensor = self.gripper_angle_signs.clone() * self.gripper_angle_open
        self.gripper_angle_close_tensor = self.gripper_angle_signs.clone() * self.gripper_angle_close

    def _set_sim_params(self, args):
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 40.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        if args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 24
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.001 / 4
            sim_params.physx.contact_offset = 0.002 / 4  # TODO: check the comparison between 0.001 and 0.002
            sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP
            sim_params.physx.bounce_threshold_velocity = 0.2
            sim_params.physx.max_depenetration_velocity = 10
            sim_params.physx.friction_offset_threshold = 0.004 / 4
            sim_params.physx.friction_correlation_distance = 0.0025 / 4
            sim_params.physx.num_threads = 4
            sim_params.physx.use_gpu = args.use_gpu
        else:
            raise Exception("Only tested with PhysX")

        return sim_params

    # def is_static(self, env_ids):
    #     return torch.all(self.block_state[env_ids, :, 7:10].abs() < 0.02)

    # def wait_static(self, env_ids, max_count=20, static_threshold=0):
    #     static_count = 0
    #     total_count = 0
    #     while total_count < max_count:
    #         self.step(env_ids)
    #         if self.is_static(env_ids):
    #             static_count += 1
    #         else:
    #             static_count = 0
    #         if static_count > static_threshold:
    #             return True
    #         total_count += 1
    #     print(self.block_state[env_ids, :, 7:10].abs().max())
    #     print(f"Warning: objects exceeded {total_count}/{max_count} steps.")
    #     return False

    def is_static(self, env_ids, prev_pos_state, prev_ang_state, threshold=0.002, angle_threshold=0.05):
        return torch.logical_and(
            torch.all((prev_pos_state - self.block_state[env_ids, :, 0:3]).abs() < threshold, dim=2),
            torch.all((prev_ang_state - self.block_state[env_ids, :, 3:7]).abs() < angle_threshold, dim=2),
        )
        # return (prev_pos_state - self.block_state[env_ids, :, 0:3]).abs() < threshold

    def wait_static(self, env_ids, max_count=15, static_threshold=1, is_real=False):
        # if IS_REAL:
        #     max_count = 20
        # if is_real:
        #     max_count *= 2
        static_count = 0
        total_count = 0
        while total_count < max_count:
            prev_pos_state = self.block_state[env_ids, :, 0:3].clone()
            prev_ang_state = self.block_state[env_ids, :, 3:7].clone()
            self.step(env_ids)
            # if is_real:
            #     static_state = self.is_static(env_ids, prev_pos_state, prev_ang_state, threshold=0.01, angle_threshold=0.1)
            # else:
            #     static_state = self.is_static(env_ids, prev_pos_state, prev_ang_state)
            static_state = self.is_static(env_ids, prev_pos_state, prev_ang_state)
            if torch.all(static_state):
                static_count += 1
            else:
                static_count = 0
            if static_count > static_threshold:
                return []
            total_count += 1
        # static_state = torch.reshape(static_state, (static_state.shape[0], -1))
        non_static_idx = (torch.all(static_state, dim=1) == 0).nonzero().flatten()
        print(
            Fore.YELLOW
            + f"Warning: objects exceeded {total_count}/{max_count} steps. {(prev_pos_state - self.block_state[env_ids, :, 0:3]).abs().max()}. {(prev_ang_state - self.block_state[env_ids, :, 3:7]).abs().max()}. {non_static_idx}."
        )
        return non_static_idx

    def add_object_from_file(self, file_name):

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

        # asset_root = "../../assets/urdf/blocks/"  # TODO: change this path to assets/urdf, which should be realistic
        asset_root = "assets/blocks/"
        block_assets = []
        unique_models = {}
        block_names = []
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.override_com = True
        asset_options.override_inertia = True
        for object_idx in range(num_obj):
            block_file = block_files[object_idx]
            if block_file in unique_models:
                block_assets.append(unique_models[block_file])
            else:
                if "concave" in block_file:
                    asset_options.vhacd_enabled = True
                    asset_options.vhacd_params.resolution = 64000000
                    asset_options.vhacd_params.alpha = 0.005
                    asset_options.vhacd_params.beta = 0.005
                else:
                    asset_options.vhacd_enabled = False
                block_asset = self.gym.load_asset(self.sim, asset_root, block_file, asset_options)
                block_props = self.gym.get_asset_rigid_shape_properties(block_asset)
                if IS_REAL:
                    block_props[0].friction = 0.3
                else:
                    block_props[0].friction = 0.3
                # block_props[0].rolling_friction = 0.001
                # block_props[0].torsion_friction = 0.001
                self.gym.set_asset_rigid_shape_properties(block_asset, block_props)
                block_assets.append(block_asset)
                unique_models[block_file] = block_asset
            block_names.append(block_file.split(".")[0])

        self.block_names = block_names
        self.defined_actions = load_pre_defined_actions(block_names)

        return block_assets, block_poses, block_colors

    def save_object_states(self, env_ids, max_count=3):
        """Save states of all rigid objects."""
        total_count = 0
        while total_count < max_count:
            non_static_idx = self.wait_static(env_ids)

            if len(non_static_idx) == 0:
                break

            total_count += 1

        all_states = self.all_state[env_ids].clone()
        # all_states[:, :, 7:13] = 0
        if len(non_static_idx) == 0:
            return True, all_states, non_static_idx
        else:
            print(Fore.RED + f"Failed to save object states!!! {non_static_idx}")
            return False, all_states, non_static_idx

    def restore_object_states(self, env_ids, all_states, max_count=3, threshold=0.005, angle_threshold=0.1):
        """Restore states of all rigid objects."""
        total_count = 0
        all_indices = self.global_indices[env_ids].flatten()
        len_all_indices = len(all_indices)
        while total_count < max_count:
            self.all_state[env_ids, ...] = all_states
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, self.actor_root_state_tensor, gymtorch.unwrap_tensor(all_indices), len_all_indices,
            )
            non_static_idx = self.wait_static(env_ids)
            if torch.all((self.all_state[env_ids, :, 0:3] - all_states[..., 0:3]).abs() < threshold):
                if torch.all((self.all_state[env_ids, :, 3:7] - all_states[..., 3:7]).abs() < angle_threshold):
                    if len(non_static_idx) > 0:
                        print(
                            Fore.YELLOW
                            + f"Reached but failed to be static!!!  {(self.all_state[env_ids, :, 0:3] - all_states[..., 0:3]).abs().max()}. {(self.all_state[env_ids, :, 3:7] - all_states[..., 3:7]).abs().max()} {env_ids[non_static_idx]}."
                        )
                    return True, []

            total_count += 1
            # all_states[..., 7:13] = 0

        if len(non_static_idx) > 0:
            print(Fore.YELLOW + "timeout to restore object states!!!")
        angle_diff = (self.all_state[env_ids, :, 3:7] - all_states[..., 3:7]).abs()
        angle_failed = angle_diff > angle_threshold
        angle_failed = torch.reshape(angle_failed, (angle_failed.shape[0], -1))
        angle_failed = torch.any(angle_failed, dim=1)
        diff = (self.all_state[env_ids, :, 0:3] - all_states[..., 0:3]).abs()
        failed = diff >= threshold
        failed = torch.reshape(failed, (failed.shape[0], -1))
        failed = torch.any(failed, dim=1)
        failed = failed | angle_failed
        failed_idx = failed.nonzero().flatten()
        print(
            Fore.RED + f"Failed to restore object states!!!  {diff.max()}. {angle_diff.max()}. {env_ids[failed_idx]}."
        )
        # input("wait")
        # self._create_viewer()
        # while True:
        #     self.all_state[env_ids, ...] = all_states
        #     # self.all_state[env_ids, :, 7:13] = 0
        #     self.gym.set_actor_root_state_tensor_indexed(
        #         self.sim, self.actor_root_state_tensor, gymtorch.unwrap_tensor(all_indices), len_all_indices,
        #     )
        #     non_static_idx = self.wait_static(env_ids)
        #     diff = (self.all_state[env_ids, :, 0:3] - all_states[..., 0:3]).abs()
        #     failed = diff >= threshold
        #     failed = torch.reshape(failed, (failed.shape[0], -1))
        #     failed_idx = torch.any(failed, dim=1).nonzero().flatten()
        #     print(Fore.RED + f"Failed to restore object states!!!  {diff.max()}. {env_ids[failed_idx]}.")
        return False, failed_idx

    def reset_idx(self, env_ids, is_real=False):

        # reset UR5e
        self.ur5e_dof_pos[env_ids, :] = self.ur5e_home_dof_pos
        self.ur5e_dof_vel[env_ids, :] = 0
        self.ur5e_dof_targets[env_ids, :] = self.ur5e_home_dof_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.ur5e_dof_targets),
            gymtorch.unwrap_tensor(self.global_ur5e_indices[env_ids]),
            len(self.global_ur5e_indices[env_ids]),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.dof_state_tensor,
            gymtorch.unwrap_tensor(self.global_ur5e_indices[env_ids]),
            len(self.global_ur5e_indices[env_ids]),
        )

        # reset blocks
        block_indices = self.global_indices[env_ids, 2:].flatten()
        self.block_state[env_ids, :] = self.default_block_state[env_ids, :]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.actor_root_state_tensor, gymtorch.unwrap_tensor(block_indices), len(block_indices),
        )

        non_state_idx = self.wait_static(env_ids, is_real=is_real)
        assert len(non_state_idx) == 0

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def open_gripper_idx(self, env_ids):
        self._set_gripper_target_idx(env_ids, self.gripper_angle_open_tensor)

    def close_gripper_idx(self, env_ids):
        self._set_gripper_target_idx(env_ids, self.gripper_angle_close_tensor)

    def _set_gripper_target_idx(self, env_ids, angle_tensor):
        self.ur5e_dof_targets[env_ids, 6:] = angle_tensor

        moving = torch.tensor([True] * len(env_ids), device=self.device)
        while torch.any(moving):
            self.step(env_ids)

            joints_err = self.ur5e_dof_targets[env_ids, 6:] - self.ur5e_dof_pos[env_ids, 6:]
            reach = torch.all(torch.abs(joints_err) < 0.1, dim=1)
            moving[reach] = False

    def go_home_idx(self, env_ids):
        self.ur5e_dof_targets[env_ids, :] = self.ur5e_home_dof_pos

        moving = torch.tensor([True] * len(env_ids), device=self.device)
        while torch.any(moving):
            self.step(env_ids)

            joints_err = self.ur5e_dof_targets[env_ids, :] - self.ur5e_dof_pos[env_ids, :]
            reach = torch.all(torch.abs(joints_err) < 0.1, dim=1)
            moving[reach] = False

    def set_target_joints_idx(self, env_ids, target_pos):
        # move the joints of UR5e
        self.ur5e_dof_targets[env_ids, :6] = target_pos

        # maintain the gripper
        # finger_joint = self.ur5e_dof_targets[env_ids, 6:7]
        # gripper_joints = finger_joint * self.gripper_angle_signs
        # self.ur5e_dof_targets[env_ids, 6:] = gripper_joints

    def set_target_joints_with_gripper_idx(self, env_ids, target_pos, angle_tensor):
        # move the joints of UR5e
        self.ur5e_dof_targets[env_ids, :6] = target_pos

        # maintain the gripper
        # gripper_joints = self.gripper_main_joint_pos[env_ids] * self.gripper_angle_signs
        # gripper_joints[env_ids, 0] = angle_tensor[env_ids, 0]
        gripper_joints = self.gripper_main_joint_pos[env_ids] * self.gripper_angle_signs
        gripper_joints[:, 0] += torch.clamp(angle_tensor[:, 0] - self.gripper_main_joint_pos[env_ids, 0], -0.02, 0.02)
        self.ur5e_dof_targets[env_ids, 6:] = gripper_joints

    def control_ik_idx(self, env_ids, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef[env_ids], 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef[env_ids] @ j_eef_T + lmbda) @ dpose).view(len(env_ids), 6)
        return u

    def ik(self, P: torch.Tensor) -> torch.Tensor:
        # input: batch_size x 4 (x, y, z, theta)
        # output: batch_size x 6
        batch_size = P.shape[0]
        x = -P[..., 1]
        y = -P[..., 0]
        h = P[..., 2] - self.ik_d0 + self.ik_d5 + self.ik_offset
        theta = P[..., 3]
        tmp1 = torch.sqrt(x * x + y * y - self.ik_d3_2)
        tmp = torch.hypot(tmp1 - self.ik_d4, h)
        y = -y

        theta1 = -torch.atan(x / y) + torch.atan(self.ik_d3 / tmp1)
        theta2 = (
            -np.pi
            + torch.asin(h / tmp)
            + torch.acos(-(self.ik_a1_2 + tmp * tmp - self.ik_a2_2) / (2 * self.ik_a1 * tmp))
        )
        theta3 = -np.pi + torch.acos((self.ik_a1_2 + self.ik_a2_2 - tmp * tmp) / (2 * self.ik_a1_a2))
        theta4 = (
            -np.pi
            + torch.acos(h / tmp)
            + torch.acos(-(self.ik_a2_2 + tmp * tmp - self.ik_a1_2) / (2 * self.ik_a2 * tmp))
        )
        theta5 = np.pi / 2 * torch.ones(batch_size, device=self.device)
        theta1[theta1 > np.pi] -= 2 * np.pi
        theta6 = theta1 + theta + np.pi / 2

        return torch.stack((theta1, theta2, theta3, theta4, theta5, theta6), dim=1)

    def straight_move_idx(self, num_sets, pose0, pose1, yaw, detect_force=False, distance=0.01):
        """Move the end effector in a straight line, as a helper of push_idx function"""

        step_distance = distance  # every step_distance cm
        vec = pose1 - pose0
        length = torch.linalg.norm(vec, dim=1, keepdim=True)
        vec = vec / length * step_distance
        n_push = torch.floor(length / step_distance).view(num_sets, 1)  # every 1 cm
        max_n_push = torch.max(n_push).int()

        poses = torch.zeros(num_sets, max_n_push + 1, 4, device=self.device)
        poses[:, 0, :3] = pose0 + vec
        for n in range(1, max_n_push):
            poses[:, n, :3] = torch.where(n < n_push, poses[:, n - 1, :3] + vec, poses[:, n - 1, :3])
        poses[:, -1, :3] = pose1
        poses[:, :, 3] = yaw

        if detect_force:
            detect_force_signs = torch.tensor([True] * (max_n_push + 1) * num_sets, device=self.device).view(
                num_sets, max_n_push + 1
            )
        else:
            detect_force_signs = torch.tensor([False] * (max_n_push + 1) * num_sets, device=self.device).view(
                num_sets, max_n_push + 1
            )

        return poses, detect_force_signs

    def grasp_idx(self, env_ids, pose, angle):
        """Execute grasping primitive.
        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle
        Returns:
            success: robot movement success if True.
        """
        # s = time.time()
        # Open the gripper
        self.open_gripper_idx(env_ids)

        # Adjust grasp positions.
        pose = to_torch(pose, device=self.device)
        pose[:, 2] += self.gripper_center_offset - 0.01
        angle = to_torch(angle, device=self.device)

        # Align against grasp direction.
        yaw = ((angle) % np.pi) - np.pi / 2
        yaw[yaw > np.pi / 2] = yaw[yaw > np.pi / 2] - np.pi
        yaw[yaw < -np.pi / 2] = np.pi + yaw[yaw < -np.pi / 2]

        # Add intermediate poses
        over = torch.clone(pose)
        over[:, 2] += 0.1

        # Compute and combine push path, TODO: this part can be parallelized
        num_sets = len(env_ids)
        # -> over
        pose_paths = [
            torch.cat((torch.clone(over).view(num_sets, 1, 3), yaw.view(num_sets, 1, 1)), dim=2),
        ]
        pose_path_detect_force_signs = [
            torch.tensor([False] * num_sets, device=self.device).unsqueeze(-1),
        ]
        pose_gripper_angle = [self.gripper_angle_open_tensor.repeat(num_sets, 1, 1).view(num_sets, 1, -1)]
        # over -> pose
        poses, detect_force_signs = self.straight_move_idx(num_sets, over, pose, yaw, detect_force=True)
        pose_paths.append(poses)
        pose_path_detect_force_signs.append(detect_force_signs)
        pose_gripper_angle.append(
            self.gripper_angle_open_tensor.repeat(num_sets, poses.shape[1], 1).view(num_sets, poses.shape[1], -1)
        )
        # close gripper
        pose_paths.append(poses[:, -1:, :])
        no_detect = torch.zeros_like(detect_force_signs[:, -1:], device=self.device)
        no_detect[:, :] = False
        pose_path_detect_force_signs.append(no_detect)
        pose_gripper_angle.append(self.gripper_angle_close_tensor.repeat(num_sets, 1, 1).view(num_sets, 1, -1))
        # pose -> over
        poses, detect_force_signs = self.straight_move_idx(num_sets, pose, over, yaw, detect_force=False)
        pose_paths.append(poses)
        pose_path_detect_force_signs.append(detect_force_signs)
        pose_gripper_angle.append(self.gripper_angle_close_tensor.expand(num_sets, -1).view(num_sets, 1, -1))
        pose_gripper_angle.append(
            self.gripper_angle_close_tensor.repeat(num_sets, poses.shape[1], 1).view(num_sets, poses.shape[1], -1)
        )

        # group
        pose_paths = torch.cat(pose_paths, dim=1)
        pose_path_detect_force_signs = torch.cat(pose_path_detect_force_signs, dim=1)
        pose_gripper_angle = torch.cat(pose_gripper_angle, dim=1)

        # IK
        joint_paths = self.ik(pose_paths.view(-1, 4)).view(num_sets, -1, 6)

        # Execute push
        gripper_static_count = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        # prev_gripper_pos = self.gripper_main_joint_pos[env_ids].clone()
        prev_gripper_pos = self.ur5e_dof_pos[env_ids, 6:].clone()
        joint_static_count = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        prev_joint_pos = self.ur5e_dof_pos[env_ids, :6].clone()
        moving = torch.tensor([True] * num_sets, device=self.device)
        force_record = torch.tensor([False] * num_sets, device=self.device)
        step_ids = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        local_env_ids = torch.arange(num_sets, dtype=torch.long, device=self.device)
        num_step = pose_paths.shape[1]
        # max_sim_step = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        while torch.any(moving):
            # step
            target_pos = joint_paths[local_env_ids, step_ids, :]
            gripper_target_pos = pose_gripper_angle[local_env_ids, step_ids, :]
            self.set_target_joints_with_gripper_idx(env_ids, target_pos, gripper_target_pos)
            self.step(env_ids)

            # go to the next target joints; change moving status, 0 means idle, 1 means moving
            joints_err = target_pos - self.ur5e_dof_pos[env_ids, :6]
            detect_force = (
                torch.logical_or(self.force_l_z[env_ids].abs() > 50, self.force_r_z[env_ids].abs() > 50)
                & pose_path_detect_force_signs[local_env_ids, step_ids]
            )
            reach = torch.all(torch.abs(joints_err) < 0.02, dim=1)
            joint_static = torch.all(
                torch.abs(self.ur5e_dof_pos[env_ids, :6] - prev_joint_pos) < 0.02, dim=1
            ).squeeze_()
            joint_static_count[joint_static] += 1
            joint_static_count[torch.logical_not(joint_static)] = 0
            reach = torch.logical_or(reach, joint_static_count > 10)

            gripper_static = torch.all((torch.abs(self.ur5e_dof_pos[env_ids, 6:] - prev_gripper_pos) < 0.01), dim=1)
            gripper_static_count[gripper_static] += 1
            gripper_static_count[torch.logical_not(gripper_static)] = 0
            # gripper_reach = gripper_static_count > 10
            gripper_reach = torch.logical_or(
                gripper_static_count > 50,
                torch.logical_or(
                    self.force_l_y[env_ids].abs() > 50, self.force_r_y[env_ids].abs() > 50
                ),  # TODO: this 50 can be tuned
            )

            reach = torch.logical_and(reach, gripper_reach)
            step_ids[reach] += 1
            # ====
            # max_sim_step += 1
            # should_reset = (max_sim_step > 100).clone()
            # step_ids[should_reset] += 1
            # max_sim_step[should_reset] = 0
            # ====
            complete = step_ids >= num_step
            step_ids[torch.logical_or(complete, detect_force)] = num_step - 1
            moving[complete] = False
            force_record |= detect_force

            prev_joint_pos = self.ur5e_dof_pos[env_ids, :6].clone()
            # prev_gripper_pos = self.gripper_main_joint_pos[env_ids].clone()
            prev_gripper_pos = self.ur5e_dof_pos[env_ids, 6:].clone()

        is_close = (self.gripper_main_joint_pos[env_ids] < self.gripper_angle_close_threshold).squeeze_()
        # e = time.time()
        # print(f"grasp time {e-s}")
        return torch.logical_and(torch.logical_not(force_record), is_close)

    def push_idx(self, env_ids, pose0, pose1):
        """Execute pushing primitive.
        Args:
            env_ids: index of env should push
            pose0: array of SE(3) starting pose.
            pose1: array of SE(3) ending pose.
        Returns:
            success: robot movement success if True.
        """
        # s = time.time()
        # Close the gripper
        self.close_gripper_idx(env_ids)

        # Adjust push start and end positions
        pose0 = to_torch(pose0, device=self.device)
        pose1 = to_torch(pose1, device=self.device)
        pose0[:, 2] += self.gripper_center_offset
        pose1[:, 2] += self.gripper_center_offset

        # Align against push direction
        vec = pose1 - pose0
        yaw = -torch.atan2(vec[:, 1], vec[:, 0]).view(len(env_ids), 1)
        yaw[yaw > np.pi / 2] = yaw[yaw > np.pi / 2] - np.pi
        yaw[yaw < -np.pi / 2] = np.pi + yaw[yaw < -np.pi / 2]

        # Add intermediate poses
        over0 = torch.clone(pose0)
        over0[:, 2] += 0.1
        length = torch.linalg.norm(vec, dim=1, keepdim=True)
        vec = vec / length * 0.02  # back 2 cm
        over1 = torch.clone(pose1)
        over1 = over1 - vec
        over1[:, 2] += 0.1

        # Compute and combine push path, TODO: this part can be parallelized
        num_sets = len(env_ids)
        # -> over0
        pose_paths = [
            torch.cat((torch.clone(over0).view(num_sets, 1, 3), yaw.view(num_sets, 1, 1)), dim=2),
        ]
        pose_path_detect_force_signs = [
            torch.tensor([False] * num_sets, device=self.device).unsqueeze(-1),
        ]
        # over0 -> pose0
        poses, detect_force_signs = self.straight_move_idx(num_sets, over0, pose0, yaw, detect_force=True)
        pose_paths.append(poses)
        pose_path_detect_force_signs.append(detect_force_signs)
        # pose0 -> pose1
        poses, detect_force_signs = self.straight_move_idx(num_sets, pose0, pose1, yaw, detect_force=False)
        pose_paths.append(poses)
        pose_path_detect_force_signs.append(detect_force_signs)
        # pose1 -> over1
        poses, detect_force_signs = self.straight_move_idx(num_sets, pose1, over1, yaw, detect_force=False)
        pose_paths.append(poses)
        pose_path_detect_force_signs.append(detect_force_signs)

        # group
        pose_paths = torch.cat(pose_paths, dim=1)
        pose_path_detect_force_signs = torch.cat(pose_path_detect_force_signs, dim=1)

        # IK
        joint_paths = self.ik(pose_paths.view(-1, 4)).view(num_sets, -1, 6)

        # Execute push
        target_gripper_pos = self.ur5e_dof_targets[env_ids, 6:].clone()
        prev_gripper_pos = self.ur5e_dof_pos[env_ids, 6:].clone()
        gripper_static_count = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        prev_joint_pos = self.ur5e_dof_pos[env_ids, :6].clone()
        joint_static_count = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        moving = torch.tensor([True] * num_sets, device=self.device)
        step_ids = torch.zeros(num_sets, dtype=torch.long, device=self.device)
        local_env_ids = torch.arange(num_sets, dtype=torch.long, device=self.device)
        num_step = pose_paths.shape[1]
        success_state = torch.ones(num_sets, dtype=torch.bool, device=self.device)
        while torch.any(moving):
            # step
            target_pos = joint_paths[local_env_ids, step_ids, :]
            self.set_target_joints_idx(env_ids, target_pos)
            self.step(env_ids)

            # go to the next target joints; change moving status, 0 means idle, 1 means moving
            joints_err = target_pos - self.ur5e_dof_pos[env_ids, :6]
            detect_force = (
                torch.logical_or(self.force_l_z[env_ids].abs() > 200, self.force_r_z[env_ids].abs() > 200)
                & pose_path_detect_force_signs[local_env_ids, step_ids]
            )
            reach = torch.all(torch.abs(joints_err) < 0.012, dim=1)  # change this to make the robot run faster
            joint_static = torch.all(
                torch.abs(self.ur5e_dof_pos[env_ids, :6] - prev_joint_pos) < 0.01, dim=1
            ).squeeze_()
            joint_static_count[joint_static] += 1
            joint_static_count[torch.logical_not(joint_static)] = 0
            reach = torch.logical_or(reach, joint_static_count > 100)

            gripper_reach = torch.all((torch.abs(self.ur5e_dof_pos[env_ids, 6:] - target_gripper_pos) < 0.012), dim=1)
            gripper_static = torch.all(torch.abs(self.ur5e_dof_pos[env_ids, 6:] - prev_gripper_pos) < 0.01, dim=1)
            gripper_static_count[gripper_static] += 1
            gripper_static_count[torch.logical_not(gripper_static)] = 0
            gripper_reach = torch.logical_or(gripper_reach, gripper_static_count > 100)

            success_state &= torch.logical_not(detect_force)
            # if torch.any(detect_force):
            #     test_ids = torch.nonzero(detect_force).flatten()
            #     print(env_ids[test_ids])
            #     print(self.force_l_z[env_ids[test_ids]], self.force_r_z[env_ids[test_ids]])
            #     print(pose0[test_ids])
            #     print(pose1[test_ids])
            #     if self.viewer is None:
            #         self._create_viewer()
            #     input("force detected, press to continue")
            #     for _ in range(1000):
            #         self.step(env_ids)
            #         print(self.force_l_z[env_ids[test_ids]], self.force_r_z[env_ids[test_ids]])
            #     # raise Exception("Force detected")
            #     # detect_force[test_ids] = False

            reach = torch.logical_and(reach, gripper_reach)
            step_ids[reach] += 1
            complete = step_ids >= num_step
            step_ids[torch.logical_or(complete, detect_force)] = num_step - 1
            moving[complete] = False

            prev_gripper_pos = self.ur5e_dof_pos[env_ids, 6:].clone()
            prev_joint_pos = self.ur5e_dof_pos[env_ids, :6].clone()

            # if torch.any(detect_force):
            #     test_ids = torch.nonzero(detect_force).flatten()
            #     print(env_ids[test_ids])
            #     print(self.force_l_z[env_ids[test_ids]], self.force_r_z[env_ids[test_ids]])
            #     print(pose0[test_ids])
            #     print(pose1[test_ids])
            #     if self.viewer is None:
            #         self._create_viewer()
            #     input("force detected, press to continue")
            #     for _ in range(500):
            #         self.step(env_ids)
            #         print(self.force_l_z[env_ids[test_ids]], self.force_r_z[env_ids[test_ids]])
            #     # raise Exception("Force detected")
        return success_state

    # def straight_move_idx(self, num_sets, pose0, pose1, detect_force=False):
    #     """Move the end effector in a straight line, as a helper of push_idx function"""

    #     step_distance = 0.01  # every step_distance cm
    #     vec = pose1 - pose0
    #     length = torch.linalg.norm(vec, dim=1, keepdim=True)
    #     vec = vec / length * step_distance
    #     n_push = torch.floor(length / step_distance).view(num_sets, 1)  # every 1 cm
    #     max_n_push = torch.max(n_push).int()

    #     poses = torch.zeros(num_sets, max_n_push + 1, 3, device=self.device)
    #     poses[:, 0, :] = pose0 + vec
    #     for n in range(1, max_n_push):
    #         poses[:, n, :] = torch.where(n < n_push, poses[:, n - 1, :] + vec, poses[:, n - 1, :])
    #     poses[:, -1, :] = pose1

    #     if detect_force:
    #         detect_force_signs = torch.tensor([True] * (max_n_push + 1) * num_sets, device=self.device).view(
    #             num_sets, max_n_push + 1
    #         )
    #     else:
    #         detect_force_signs = torch.tensor([False] * (max_n_push + 1) * num_sets, device=self.device).view(
    #             num_sets, max_n_push + 1
    #         )

    #     return poses, detect_force_signs

    # def push_idx(self, env_ids, pose0, pose1):
    #     """Execute pushing primitive.
    #     Args:
    #         env_ids: index of env should push
    #         pose0: array of SE(3) starting pose.
    #         pose1: array of SE(3) ending pose.
    #     Returns:
    #         success: robot movement success if True.
    #     """

    #     # Close the gripper
    #     self.close_gripper_idx(env_ids)

    #     # Adjust push start and end positions
    #     pose0 = to_torch(pose0, device=self.device)
    #     pose1 = to_torch(pose1, device=self.device)
    #     pose0[:, 2] += self.gripper_center_offset
    #     pose1[:, 2] += self.gripper_center_offset

    #     # Align against push direction
    #     vec = pose1 - pose0
    #     roll = torch.ones(len(env_ids), device=self.device) * np.pi
    #     pitch = torch.ones(len(env_ids), device=self.device) * 0
    #     yaw = torch.atan2(vec[:, 1], vec[:, 0])
    #     rot = quat_from_euler_xyz(roll, pitch, yaw)
    #     yaw_test = yaw.view(len(env_ids), -1)

    #     # Add intermediate poses
    #     over0 = torch.clone(pose0)
    #     over0h = torch.clone(pose0)
    #     over0[:, 2] += 0.05
    #     over0h[:, 2] += 0.1
    #     length = torch.linalg.norm(vec, dim=1, keepdim=True)
    #     vec = vec / length * 0.02  # back 2 cm
    #     over1 = torch.clone(pose1)
    #     over1[:, :] = over1 - vec
    #     over1h = torch.clone(over1)
    #     over1[:, 2] += 0.05
    #     over1h[:, 2] += 0.1

    #     # Compute and combine push path
    #     num_sets = len(env_ids)
    #     # -> over0h
    #     pose_paths = [
    #         torch.clone(over0h.view(num_sets, 1, 3)),
    #     ]
    #     pose_path_detect_force_signs = [
    #         torch.tensor([False] * num_sets, device=self.device).unsqueeze(-1),
    #     ]
    #     # over0h -> over0
    #     poses, detect_force_signs = self.straight_move_idx(num_sets, over0h, over0, detect_force=True)
    #     pose_paths.append(poses)
    #     pose_path_detect_force_signs.append(detect_force_signs)
    #     # over0 -> pose0
    #     poses, detect_force_signs = self.straight_move_idx(num_sets, over0, pose0, detect_force=True)
    #     pose_paths.append(poses)
    #     pose_path_detect_force_signs.append(detect_force_signs)
    #     # pose0 -> pose1
    #     poses, detect_force_signs = self.straight_move_idx(num_sets, pose0, pose1, detect_force=True)
    #     pose_paths.append(poses)
    #     pose_path_detect_force_signs.append(detect_force_signs)
    #     # pose1 -> over1
    #     poses, detect_force_signs = self.straight_move_idx(num_sets, pose1, over1, detect_force=False)
    #     pose_paths.append(poses)
    #     pose_path_detect_force_signs.append(detect_force_signs)
    #     # over1 -> over1h
    #     poses, detect_force_signs = self.straight_move_idx(num_sets, over1, over1h, detect_force=False)
    #     pose_paths.append(poses)
    #     pose_path_detect_force_signs.append(detect_force_signs)

    #     pose_paths = torch.cat(pose_paths, dim=1)
    #     pose_path_detect_force_signs = torch.cat(pose_path_detect_force_signs, dim=1)
    #     num_step = pose_paths.size()[1]

    #     # Execute push
    #     moving = torch.tensor([True] * num_sets, device=self.device)
    #     idle = torch.tensor([False] * num_sets, device=self.device)
    #     step_ids = torch.zeros(num_sets, dtype=torch.long, device=self.device)
    #     local_env_ids = torch.arange(num_sets, dtype=torch.long, device=self.device)

    #     while torch.any(moving):
    #         # compute position and orientation error
    #         pos_err = pose_paths[moving, step_ids[moving], :] - self.gripper_center_pos[env_ids[moving]]
    #         orn_err = orientation_error(rot[moving], self.gripper_center_rot[env_ids[moving]])

    #         # step
    #         dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    #         step_pose = self.control_ik_idx(env_ids[moving], dpose)
    #         target_pose = self.ur5e_dof_pos[env_ids[moving], :6] + step_pose
    #         # dpose = torch.cat([pose_paths[moving, step_ids[moving], :], yaw_test[moving]], 1)
    #         # target_pose = self.ik(dpose)
    #         self.set_target_joints_idx(env_ids[moving], target_pose)
    #         self.step(env_ids[moving])

    #         # go to the next target joints; change moving status, 0 means idle, 1 means moving
    #         detect_force = torch.logical_and(
    #             torch.any(self.force[env_ids[moving]].abs() > 20, dim=1),
    #             pose_path_detect_force_signs[local_env_ids, step_ids],
    #         )
    #         step_ids[moving] = torch.where(
    #             torch.all(torch.abs(pos_err) < 1e-2, dim=1), step_ids[moving] + 1, step_ids[moving]
    #         )  # TODO: check this torch.all
    #         step_ids = torch.clamp(step_ids, 0, num_step - 1)
    #         moving = torch.where(detect_force, idle, moving)
    #         moving = torch.where(step_ids == num_step - 1, idle, moving)

    def step(self, env_ids):
        # pre physics step
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.ur5e_dof_targets),
            gymtorch.unwrap_tensor(self.global_ur5e_indices[env_ids]),
            len(self.global_ur5e_indices[env_ids]),
        )

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer:
            self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def render_camera(self, env_ids, color=True, depth=False, segm=False, focal_target=False):
        """Render images for envs. If focal_target is True, then, only a small region centered on the target object will be return for depth image"""
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        color_images = []
        depth_images = []
        segm_images = []
        for i in env_ids:
            if color:
                color_image = self.gym.get_camera_image(self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(224, 224, 4)[:, :, :3]
                color_images.append(color_image)

            if depth:
                depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH)
                depth_image[depth_image == -np.inf] = 0
                depth_image += -np.min(depth_image)
                if focal_target:
                    center = self.all_state[i, self.target_handle, 0:2].clone().cpu().numpy()
                    x = int(round((center[0] - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE))
                    y = int(round((center[1] - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE))
                    depth_image = depth_image[
                        max(0, x - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, x + IMAGE_OBJ_CROP_SIZE // 2),
                        max(0, y - IMAGE_OBJ_CROP_SIZE // 2) : min(IMAGE_SIZE, y + IMAGE_OBJ_CROP_SIZE // 2),
                    ]
                    if depth_image.shape[0] < IMAGE_OBJ_CROP_SIZE or depth_image.shape[1] < IMAGE_OBJ_CROP_SIZE:
                        pad = [[0, 0], [0, 0], [0, 0]]
                        if x - IMAGE_OBJ_CROP_SIZE // 2 < 0:
                            pad[0][0] = IMAGE_OBJ_CROP_SIZE // 2 - x
                        if y - IMAGE_OBJ_CROP_SIZE // 2 < 0:
                            pad[1][0] = IMAGE_OBJ_CROP_SIZE // 2 - y
                        if x + IMAGE_OBJ_CROP_SIZE // 2 > IMAGE_SIZE:
                            pad[0][1] = (x + IMAGE_OBJ_CROP_SIZE // 2) - IMAGE_SIZE
                        if y + IMAGE_OBJ_CROP_SIZE // 2 > IMAGE_SIZE:
                            pad[1][1] = (y + IMAGE_OBJ_CROP_SIZE // 2) - IMAGE_SIZE
                        depth_image = np.pad(depth_image, pad[:2], "constant", constant_values=0)
                depth_images.append(depth_image)

            # assume the first object is the target
            if segm:
                segm_image = self.gym.get_camera_image(
                    self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_SEGMENTATION
                )
                segm_image = segm_image.astype(np.uint8)
                segm_images.append(segm_image)

        return color_images, depth_images, segm_images

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()

            # step graphics
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def set_args():
    custom_parameters = [
        {
            "name": "--controller",
            "type": str,
            "default": "ik",
            "help": "Controller to use for Franka. Options are {ik, osc}",
        },
        {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
        {"name": "--test_case", "type": str, "default": "test-cases/hard/test01.txt", "help": "Test case to create"},
    ]
    args = gymutil.parse_arguments(description="UR5e MCTS", custom_parameters=custom_parameters, headless=True)

    return args


if __name__ == "__main__":
    # set random seed
    np.random.seed(1234)

    torch.set_printoptions(precision=4, sci_mode=False)

    args = set_args()

    env = Environment(args)

    env_ids = torch.arange(args.num_envs, device=env.device)
    env.wait_static(env_ids)

    # states = env.save_object_states(env_ids).cpu().numpy()[0][2:]
    # defined_actions = env.defined_actions
    # env_ids = torch.arange(1, device=env.device)
    # color_images, depth_images, segm_images = env.render_camera(env_ids, True, True, True)
    # s = time.time()
    # for _ in range(1):
    #     sample_pre_defined_actions(color_images[0], defined_actions, states, plot=True)
    # e = time.time()
    # print(e - s)
    # s = time.time()
    # for _ in range(1):
    #     sample_actions(color_images[0], segm_images[0], plot=True)
    # e = time.time()
    # print(e - s)
    # pool = Pool(8)
    # s = time.time()
    # for _ in range(10):
    #     sample_actions_parallel(color_images[0], segm_images[0], pool=pool, plot=False)
    # e = time.time()
    # print(e - s)
    # s = time.time()
    # for _ in range(10):
    #     sample_pre_defined_actions_parallel(color_images[0], defined_actions, states, pool=pool, plot=False)
    # e = time.time()
    # print(e - s)

    state = torch.tensor(
        [
            [
                [
                    5.0000e-01,
                    0.0000e00,
                    5.0000e-04,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    5.0911e-01,
                    6.3972e-03,
                    2.3999e-02,
                    -2.8951e-05,
                    8.9917e-05,
                    2.4773e-01,
                    9.6883e-01,
                    -3.6417e-04,
                    5.6184e-05,
                    -6.9366e-04,
                    -1.8242e-02,
                    -2.0385e-02,
                    2.9017e-02,
                ],
                [
                    5.1958e-01,
                    -1.8245e-02,
                    2.4000e-02,
                    5.6504e-05,
                    5.2445e-06,
                    2.1792e-01,
                    9.7597e-01,
                    -1.6508e-04,
                    3.5452e-04,
                    -1.0804e-04,
                    -1.4943e-02,
                    -2.7954e-03,
                    3.8161e-03,
                ],
                [
                    4.9628e-01,
                    2.6728e-02,
                    2.3995e-02,
                    4.0693e-05,
                    1.2867e-04,
                    -9.6166e-01,
                    2.7426e-01,
                    5.3260e-04,
                    -6.9545e-04,
                    -7.2841e-04,
                    4.0804e-02,
                    -8.6852e-03,
                    -9.0634e-03,
                ],
                [
                    5.1782e-01,
                    -7.3856e-02,
                    2.4000e-02,
                    1.0524e-07,
                    -1.4045e-07,
                    8.0394e-01,
                    5.9470e-01,
                    1.4921e-05,
                    4.9802e-05,
                    -1.9743e-05,
                    -2.6344e-04,
                    1.7286e-04,
                    -1.1563e-05,
                ],
                [
                    4.2427e-01,
                    -3.0228e-02,
                    2.4000e-02,
                    -2.8364e-07,
                    -1.6072e-06,
                    1.3269e-01,
                    9.9116e-01,
                    8.1652e-05,
                    5.1677e-06,
                    -5.8594e-05,
                    7.7982e-05,
                    -4.9358e-04,
                    2.0107e-04,
                ],
                [
                    5.0485e-01,
                    8.8656e-02,
                    2.4000e-02,
                    -6.4029e-08,
                    2.1070e-07,
                    7.9554e-01,
                    6.0590e-01,
                    1.0243e-06,
                    3.3257e-06,
                    -1.5081e-05,
                    -4.8719e-04,
                    8.1677e-05,
                    7.6005e-07,
                ],
                [
                    5.7885e-01,
                    5.1340e-02,
                    2.4000e-02,
                    2.4942e-06,
                    -5.9906e-06,
                    2.6007e-02,
                    9.9966e-01,
                    -4.2469e-04,
                    6.8531e-05,
                    -3.7692e-04,
                    -8.9841e-04,
                    -7.1602e-03,
                    -7.3912e-03,
                ],
                [
                    5.7671e-01,
                    1.2291e-01,
                    2.4000e-02,
                    -3.5753e-06,
                    -2.7640e-06,
                    7.0885e-01,
                    7.0536e-01,
                    -9.6391e-05,
                    1.6441e-04,
                    -7.4711e-05,
                    -6.4190e-03,
                    -5.9722e-03,
                    2.3085e-04,
                ],
                [
                    4.3488e-01,
                    4.6397e-02,
                    2.4000e-02,
                    2.2904e-06,
                    5.0482e-06,
                    4.6827e-01,
                    8.8359e-01,
                    -1.8189e-05,
                    -1.2790e-04,
                    -8.0112e-05,
                    3.0793e-03,
                    5.2116e-03,
                    2.3415e-04,
                ],
                [
                    4.4541e-01,
                    -9.4959e-02,
                    2.4000e-02,
                    9.8227e-07,
                    -6.3538e-06,
                    7.9465e-01,
                    6.0706e-01,
                    4.8289e-05,
                    3.6328e-04,
                    -9.9180e-05,
                    -4.6395e-03,
                    -1.1633e-03,
                    4.5452e-04,
                ],
                [
                    5.8796e-01,
                    -4.1144e-02,
                    2.4000e-02,
                    4.7925e-06,
                    5.9661e-06,
                    6.1090e-01,
                    7.9171e-01,
                    -1.2088e-04,
                    -1.1293e-06,
                    -1.8557e-04,
                    -2.3709e-04,
                    1.8625e-03,
                    -2.4314e-04,
                ],
            ]
        ],
        device="cuda:0",
    )
    env.restore_object_states(env_ids, state)
    defined_actions = env.defined_actions
    # env_ids = torch.arange(1, device=env.device)
    # env_ids2 = torch.arange(1, 2, device=env.device)
    color_images, depth_images, segm_images = env.render_camera(env_ids, True, True, True)
    state = state[0].cpu().numpy()[2:]
    # actions = sample_pre_defined_actions(color_images[0], defined_actions, state, plot=True)
    # print(actions)
    pose0 = [[0.45, 0.106, 0.01]] * len(env_ids)
    pose1 = [[0.496, 0.088, 0.01]] * len(env_ids)
    env.push_idx(env_ids, pose0, pose1)
    # pose0 = [[0.64, -0.094, 0.01]] * len(env_ids)
    # pose1 = [[0.594, -0.092, 0.01]] * len(env_ids)
    # env.push_idx(env_ids, pose0, pose1)

    # test 01
    # env_ids = torch.arange(0, 2, device=env.device)
    # pose0 = [[0.494, -0.062, 0.01]] * len(env_ids)
    # pose1 = [[0.512, 0.036, 0.01]] * len(env_ids)
    # pose0 = [[5.040000000000000036e-01, -7.000000000000000666e-02, 0.01]] * len(env_ids)
    # pose1 = [[5.040000000000000036e-01, 2.999999999999999889e-02, 0.01]] * len(env_ids)
    # env.push_idx(env_ids, pose0, pose1)
    # pose0 = [[5.540000000000000480e-01, 9.400000000000000022e-02, 0.01]] * len(env_ids)
    # pose1 = [[4.540000000000000147e-01, 9.400000000000000022e-02, 0.01]] * len(env_ids)
    # env.push_idx(env_ids, pose0, pose1)
    # pose0 = [[4.400000000000000577e-01, 5.800000000000002376e-02, 0.01]] * len(env_ids)
    # pose1 = [[4.319689898685965979e00 + np.pi / 2]] * len(env_ids)
    # env.grasp_idx(env_ids, pose0, pose1)

    # test 02
    # env_ids = torch.arange(0, 2, device=env.device)
    # push_start = [[64 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 166 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    #     env_ids
    # )
    # push_end = [[99 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 131 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    #     env_ids
    # )
    # s = time.time()
    # env.push_idx(env_ids, push_start, push_end)
    # push_start = [[130 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 111 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    #     env_ids
    # )
    # push_end = [[81 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 121 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    #     env_ids
    # )
    # env.push_idx(env_ids, push_start, push_end)
    # # push_start = [[70 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 130 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    # #     env_ids
    # # )
    # # push_end = [[120 * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], 130 * PIXEL_SIZE + WORKSPACE_LIMITS[1][0], 0.01]] * len(
    # #     env_ids
    # # )
    # # env.push_idx(env_ids, push_start, push_end)
    # # env.wait_static(env_ids)
    # # e = time.time()
    # # print(e - s)
    # pose0 = [[4.320000000000000506e-01, -4.000000000000003553e-03, 0.01]] * len(env_ids)
    # pose1 = [[1.963495408493620697e00]] * len(env_ids)
    # env.grasp_idx(env_ids, pose0, pose1)

    # env_ids = torch.arange(1, 2, device=env.device)
    # env.restore_object_states(env_ids, object_states)
    # pose0 = [[0.494, -0.062, 0.01]] * len(env_ids)
    # pose1 = [[0.512, 0.036, 0.01]] * len(env_ids)
    # # pose0 = [[5.080000000000000071e-01, 8.999999999999999667e-02, 0.01]] * len(env_ids)
    # # pose1 = [[5.080000000000000071e-01, -1.000000000000000888e-02, 0.01]] * len(env_ids)
    # # env.push_idx(env_ids, pose0, pose1)
    # pose = [[0.559, -0.008, 0.01]] * len(env_ids)
    # angle = [[0.2]] * len(env_ids)
    # # pose = [[5.180000000000000160e-01, -1.020000000000000073e-01, 0.01]] * len(env_ids)
    # # angle = [[0.3780]] * len(env_ids)
    # # sucess = env.grasp_idx(env_ids, pose, angle)
    # # print(sucess)

    # env_ids = torch.arange(0, 4, device=env.device)
    # pose0 = [[4.899999999999999911e-01, -1.160000000000000059e-01, 0.01]] * len(env_ids)
    # pose1 = [[4.320000000000000506e-01, -3.400000000000000244e-02, 0.01]] * len(env_ids)
    # env.push_idx(env_ids, pose0, pose1)
    # pose0 = [[4.160000000000000098e-01, -1.599999999999998823e-02, 0.01]] * len(env_ids)
    # pose1 = [[2.748893571891068976e00]] * len(env_ids)
    # sucess = env.grasp_idx(env_ids, pose0, pose1)
    # print(sucess)

    # color_images, depth_images, segm_images = env.render_camera(env_ids)

    # from action_utils import sample_actions
    # from constants import PIXEL_SIZE, WORKSPACE_LIMITS

    # actions = sample_actions(color_images[0], segm_images[0], plot=True)

    # env_ids = torch.arange(0, len(actions) + 1, dtype=torch.int32, device=env.device)

    # pose0 = []
    # pose1 = []
    # for action in actions:
    #     primitive_position = [
    #         action[0][0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
    #         action[0][1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
    #         0.01,
    #     ]
    #     primitive_position_end = [
    #         action[1][0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
    #         action[1][1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
    #         0.01,
    #     ]
    #     pose0.append(primitive_position)
    #     pose1.append(primitive_position_end)
    # pose0 = torch.tensor(pose0)
    # pose1 = torch.tensor(pose1)
    # env.push_idx(env_ids, pose0, pose1)
    env_ids = torch.arange(1, device=env.device)
    color_images, depth_images, segm_images = env.render_camera(env_ids, True, True, True)
    print(len(color_images))
    for i in range(len(color_images)):
        color_image = im.fromarray(color_images[i], mode="RGB")
        color_image.save(f"test/color_env{i}.png")

        depth_image = np.round(depth_images[i] * 100000).astype(np.uint16)
        depth_image = im.fromarray(depth_image, mode="I;16")
        depth_image.save(f"test/depth_env{i}.png")

        segm_image = im.fromarray(segm_images[i], mode="L")
        segm_image.save(f"test/segm_env{i}.png")

    # test_image = np.zeros_like(depth_images[i])
    # long = 0.0225 - 0.001
    # short = 0.01125 - 0.001
    # for i in range(len(env.block_state[0])):
    #     x, y = env.block_state[0, i, 0:2]
    #     x = x.cpu().item()
    #     y = y.cpu().item()
    #     print(x, y)
    #     x0 = min(223, max(0, int(round((x - short - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE))))
    #     x1 = min(223, max(0, int(round((x + short - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE)))) + 1
    #     y0 = min(223, max(0, int(round((y - long - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE))))
    #     y1 = min(223, max(0, int(round((y + long - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE)))) + 1
    #     test_image[x0:x1, y0:y1] = 0.045
    # depth_image = np.round(test_image * 100000).astype(np.uint16)
    # depth_image = im.fromarray(depth_image, mode="I;16")
    # depth_image.save(f"test/newdepth_env.png")

    # color_image = cv2.imread("test/color_env0.png")
    # mask_image = cv2.imread("test/segm_env0.png", cv2.IMREAD_UNCHANGED)

    # pool = Pool()
    # s = time.time()
    # for i in range(10):
    #     actions = sample_actions(color_image, mask_image, plot=False)
    # e = time.time()
    # print(e - s)
    # s = time.time()
    # for i in range(10):
    #     actions = sample_actions_parallel(color_image, mask_image, pool, plot=False)
    # e = time.time()
    # print(e - s)

    # for i in range(1000):
    #     env.wait_static(env_ids)
    #     print(i)

    env.close()
