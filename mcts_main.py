"""Test"""

from collections import defaultdict
import glob
import gc
import sys
import os
import time
import datetime
import cv2
from isaacgym import gymutil
import numpy as np
from graphviz import Digraph
import random
import torch
from log_utils import Logger
import pandas as pd

from mcts.nodes import PushSearchNode
from mcts.push import PushState
from mcts.search import MonteCarloTreeSearch

# from mcts_ori.nodes import PushSearchNode
# from mcts_ori.push import PushState
# from mcts_ori.search import MonteCarloTreeSearch
from mcts_utils import MCTSHelper
from colorama import Fore, init

init(autoreset=True)
import utils
from constants import (
    GRASP_Q_PUSH_THRESHOLD,
    GRASP_Q_TEST_THRESHOLD,
    PIXEL_SIZE,
    WORKSPACE_LIMITS,
    TARGET_LOWER,
    TARGET_UPPER,
    NUM_ROTATION,
    GRASP_Q_GRASP_THRESHOLD,
)
from environment import Environment
from models import reinforcement_net, PushNet


class SeachCollector:
    def __init__(self, cases, time_limit):
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        name = ""
        for case in cases:
            name = name + case.split("/")[-1].split(".")[0] + "-"
        name = name[:-1]
        self.base_directory = os.path.join(
            os.path.abspath("logs_mcts"), "mcts-" + timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + "-" + str(time_limit) + "-" + name,
        )
        self.color_heightmaps_directory = os.path.join(self.base_directory, "data", "color-heightmaps")
        self.depth_heightmaps_directory = os.path.join(self.base_directory, "data", "depth-heightmaps")
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")
        self.prediction_directory = os.path.join(self.base_directory, "data", "predictions")
        self.visualizations_directory = os.path.join(self.base_directory, "visualizations")
        self.transitions_directory = os.path.join(self.base_directory, "transitions")
        self.executed_action_log = []
        self.label_value_log = []
        self.consecutive_log = []
        self.time_log = []
        self.planning_time_log = []
        # self.mcts_directory = os.path.join(self.base_directory, "mcts")
        # self.mcts_color_directory = os.path.join(self.base_directory, "mcts", "color")
        # self.mcts_depth_directory = os.path.join(self.base_directory, "mcts", "depth")
        # self.mcts_mask_directory = os.path.join(self.base_directory, "mcts", "mask")
        # self.mcts_child_image_directory = os.path.join(self.base_directory, "mcts", "child_image")
        self.idx = 0
        self.record_image_idx = []
        self.record_action = []
        self.record_label = []
        self.record_num_visits = []
        self.record_child_image_idx = []
        self.record_data = {
            "image_idx": self.record_image_idx,
            "action": self.record_action,
            "label": self.record_label,
            "num_visits": self.record_num_visits,
            "child_image_idx": self.record_child_image_idx,
        }

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)
        if not os.path.exists(self.prediction_directory):
            os.makedirs(self.prediction_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory))
        # if not os.path.exists(self.mcts_directory):
        #     os.makedirs(os.path.join(self.mcts_directory))
        # if not os.path.exists(self.mcts_color_directory):
        #     os.makedirs(os.path.join(self.mcts_color_directory))
        # if not os.path.exists(self.mcts_depth_directory):
        #     os.makedirs(os.path.join(self.mcts_depth_directory))
        # if not os.path.exists(self.mcts_mask_directory):
        #     os.makedirs(os.path.join(self.mcts_mask_directory))
        # if not os.path.exists(self.mcts_child_image_directory):
        #     os.makedirs(os.path.join(self.mcts_child_image_directory))

        sys.stdout = Logger(f"{self.base_directory}/log.log")
        print("Creating data logging session: %s" % (self.base_directory))

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode=0):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.color_heightmaps_directory, "%06d.%s.color.png" % (iteration, mode)), color_heightmap,
        )
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16)  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.depth_heightmaps_directory, "%06d.%s.depth.png" % (iteration, mode)), depth_heightmap,
        )

    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, "%s.log.txt" % log_name), log, delimiter=" ")

    def save_predictions(self, iteration, pred, name="push"):
        cv2.imwrite(
            os.path.join(self.prediction_directory, "%06d.png" % (iteration)), pred,
        )

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(
            os.path.join(self.visualizations_directory, "%06d.%s.png" % (iteration, name)), affordance_vis,
        )

    def _save_mcts_image(self, env, file_id, node, is_child=False):
        env.restore_objects(node.state.object_states)
        color_image, depth_image, mask_image = utils.get_true_heightmap(env)
        mask_image = utils.relabel_mask(env, mask_image)
        file_name = f"{file_id:06d}"
        if is_child:
            file_name += f"-{node.prev_move}"
        # color
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        if is_child:
            cv2.imwrite(
                os.path.join(self.mcts_child_image_directory, f"{file_name}.color.png"), color_image,
            )
        else:
            cv2.imwrite(
                os.path.join(self.mcts_color_directory, f"{file_name}.color.png"), color_image,
            )
        # depth
        depth_image = np.round(depth_image * 100000).astype(np.uint16)  # Save depth in 1e-5 meters
        if is_child:
            cv2.imwrite(
                os.path.join(self.mcts_child_image_directory, f"{file_name}.depth.png"), depth_image,
            )
        else:
            cv2.imwrite(
                os.path.join(self.mcts_depth_directory, f"{file_name}.depth.png"), depth_image,
            )
        # mask
        if is_child:
            cv2.imwrite(
                os.path.join(self.mcts_child_image_directory, f"{file_name}.mask.png"), mask_image,
            )
        else:
            cv2.imwrite(
                os.path.join(self.mcts_mask_directory, f"{file_name}.mask.png"), mask_image,
            )

        return file_name

    def save_mcts_data(self, mcts_helper, env, root, best_action, best_idx):
        _, backup_state, _ = env.save_object_states()

        search_list = [root]
        while len(search_list) > 0:
            current_node = search_list.pop(0)
            if current_node.has_children:
                save_image = False
                for i in range(len(current_node.children)):
                    child_node = current_node.children[i]
                    action = child_node.prev_move
                    # child_q = sum(sorted(child_node.q)[-MCTS_TOP:]) / min(child_node.n, MCTS_TOP)
                    # child_q = sum(child_node.q) / child_node.n
                    child_q = sum(current_node.q) / len(current_node.q)
                    self.record_image_idx.append(self.idx)
                    self.record_action.append([action.pos0[1], action.pos0[0], action.pos1[1], action.pos1[0]])
                    label = child_q
                    self.record_label.append(label)
                    self.record_num_visits.append(child_node.n)
                    child_idx = self._save_mcts_image(env, self.idx, child_node, is_child=True)
                    self.record_child_image_idx.append(child_idx)
                    save_image = True
                if save_image:
                    self._save_mcts_image(env, self.idx, current_node, is_child=False)
                    self.idx += 1
                search_list.extend(current_node.children)

        df = pd.DataFrame(self.record_data, columns=list(self.record_data.keys()))
        df.to_csv(os.path.join(self.mcts_directory, "records.csv"), index=False, header=True)

        env.restore_objects(backup_state)

    def plot_mcts(self, env, root, iteration):
        backup_state = env.save_objects()
        files = glob.glob("tree_plot/*")
        for f in files:
            os.remove(f)
        dot = Digraph(
            "mcts",
            filename=f"tree_plot/mcts{iteration}.gv",
            node_attr={"shape": "box", "fontcolor": "white", "fontsize": "3", "labelloc": "b", "fixedsize": "true",},
        )
        search_list = [root]
        while len(search_list) > 0:
            current_node = search_list.pop(0)
            node_name = current_node.state.uid
            # node_name_label = f"Q: {(sum(sorted(current_node.q)[-MCTS_TOP:]) / min(current_node.n, MCTS_TOP)):.3f},  N: {current_node.n},  Grasp Q: {current_node.state.q_value:.3f}"
            node_name_label = f"Q: {sum(current_node.q) / len(current_node.q):.3f},  N: {current_node.n},  Grasp Q: {current_node.state.q_value:.3f}"
            env.restore_objects(current_node.state.object_states)
            color_image, depth_image, _ = utils.get_true_heightmap(env)
            node_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            if current_node.prev_move is not None:
                node_action = str(current_node.prev_move).split("_")
                cv2.arrowedLine(
                    node_image,
                    (int(node_action[1]), int(node_action[0])),
                    (int(node_action[3]), int(node_action[2])),
                    (255, 0, 255),
                    2,
                    tipLength=0.4,
                )
            image_name = f"tree_plot/{node_name}.png"
            cv2.imwrite(image_name, node_image)
            depthimage_name = f"tree_plot/{node_name}-depth.png"
            depth_image = np.round(depth_image * 100000).astype(np.uint16)  # Save depth in 1e-5 meters
            cv2.imwrite(depthimage_name, depth_image)
            image_name = f"{node_name}.png"
            image_size = str(
                max(
                    0.6,
                    # sum(sorted(current_node.q)[-MCTS_TOP:]) / min(current_node.n, MCTS_TOP) * 2,
                    sum(current_node.q) / len(current_node.q) * 2,
                )
            )
            dot.node(
                node_name, label=node_name_label, image=image_name, width=image_size, height=image_size,
            )
            if current_node.parent is not None:
                node_partent_name = current_node.parent.state.uid
                dot.edge(node_partent_name, node_name)
            untracked_states = [current_node.state]
            last_node_used = False
            while len(untracked_states) > 0:
                current_state = untracked_states.pop()
                last_state_name = current_state.uid
                if last_node_used:
                    actions = current_state.get_actions()
                else:
                    if len(current_node.children) == 0:
                        actions = current_state.get_actions()
                    else:
                        actions = current_node.untried_actions
                    last_node_used = True
                for _, move in enumerate(actions):
                    # key = current_state.uid + str(move)
                    key = f"{current_state.uid}.{current_state.level}-{move}"
                    if key in current_state.mcts_helper.simulation_recorder:
                        (object_states, new_image_q,) = current_state.mcts_helper.simulation_recorder[key]
                        node_name = f"{current_state.uid}.{current_state.level}-{move}"
                        node_name_label = f"Grasp Q: {new_image_q:.3f}"
                        env.restore_objects(object_states)
                        color_image, depth_image, _ = utils.get_true_heightmap(env)
                        node_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                        node_action = str(move).split("_")
                        if len(node_action) > 1:
                            cv2.arrowedLine(
                                node_image,
                                (int(node_action[1]), int(node_action[0])),
                                (int(node_action[3]), int(node_action[2])),
                                (255, 0, 255),
                                2,
                                tipLength=0.4,
                            )
                        image_name = f"tree_plot/{node_name}.png"
                        cv2.imwrite(image_name, node_image)
                        depthimage_name = f"tree_plot/{node_name}-depth.png"
                        depth_image = np.round(depth_image * 100000).astype(np.uint16)  # Save depth in 1e-5 meters
                        cv2.imwrite(depthimage_name, depth_image)
                        image_name = f"{node_name}.png"
                        image_size = str(max(0.6, new_image_q * 2))
                        dot.node(
                            node_name, label=node_name_label, image=image_name, width=image_size, height=image_size,
                        )
                        dot.edge(last_state_name, node_name)
                        new_state, _, _, _ = current_state.move(move)
                        if new_state is not None:
                            untracked_states.append(new_state)
            search_list.extend(current_node.children)
        dot.view()
        env.restore_objects(backup_state)
        # input("wait for key")


def parse_args():
    custom_parameters = [
        {
            "name": "--controller",
            "type": str,
            "default": "ik",
            "help": "Controller to use for Franka. Options are {ik, osc}",
        },
        {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
        {"name": "--test_case", "type": str, "default": "test-cases/hard/test01.txt", "help": "Test case to create"},
        {
            "name": "--max_test_trials",
            "type": int,
            "default": 5,
            "help": "maximum number of test runs per case/scenario",
        },
        {"name": "--time_limit", "type": int, "default": 30, "help": "time allocated in second"},
        # {"name": "--switch", "type": int, "help": "Switch target"},
        {"name": "--plot", "type": bool, "default": False, "help": "Plot the MCTS"},
        {"name": "--test", "type": bool, "default": True, "help": "MCTS eval setting"},
    ]
    args = gymutil.parse_arguments(description="UR5e MCTS", custom_parameters=custom_parameters, headless=True)

    return args


if __name__ == "__main__":

    # set seed
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    iteration = 0
    args = parse_args()
    case = args.test_case
    # switch = args.switch
    test = args.test
    # if switch is not None:
    #     print(f"Target ID has been switched to {switch}")
    repeat_num = args.max_test_trials
    cases = [case] * repeat_num
    collector = SeachCollector(cases, args.time_limit)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mcts_helper = MCTSHelper("snapshot-post-020000.reinforcement.pth", "grasp_model-89.pth", seed)
    env = Environment(args)
    mcts_helper.set_env(env)
    env_ids_other = torch.arange(1, 2, device=device)
    env_ids_all = torch.arange(env.num_envs, device=device)
    env_ids_main = torch.arange(1, device=device)

    is_plot = args.plot

    num_action_log = defaultdict(list)
    for repeat_idx in range(repeat_num):
        env.reset_idx(env_ids_all)
        print(f"Reset environment at iteration {iteration} of repeat times {repeat_idx}")

        num_action = [0, 0, 0]
        planning_time = 0
        start_time = time.time()
        while True:
            num_action[0] += 1
            color_images, depth_images, _ = env.render_camera(env_ids_main, color=True, depth=True, segm=False)
            color_image = color_images[0]
            depth_image = depth_images[0]
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            print(f"Target on the table (value: {np.sum(mask) / 255}) at iteration {iteration}")
            if np.sum(mask) / 255 < 10:
                break
            q_value, best_pix_ind, grasp_predictions = mcts_helper.get_grasp_q(
                color_image, depth_image, post_checking=True
            )
            _, depth_images, _ = env.render_camera(env_ids_main, color=False, depth=True, segm=False, focal_target=True)
            new_q_value = mcts_helper.grasp_eval(depth_images[0])
            print(Fore.GREEN + f"Max grasp Q value: {q_value} / {new_q_value}")

            # record
            collector.save_heightmaps(iteration, color_image, depth_image)
            grasp_pred_vis = mcts_helper.get_prediction_vis(grasp_predictions, color_image, best_pix_ind)
            collector.save_visualizations(iteration, grasp_pred_vis, "grasp")

            # Grasp >>>>>
            if q_value > GRASP_Q_GRASP_THRESHOLD:
                best_rotation_angle = [np.deg2rad(best_pix_ind[0] * (360.0 / NUM_ROTATION)) + np.pi / 2]
                primitive_position = [
                    [
                        best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                        best_pix_ind[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                        0.01,
                    ]
                ]

                success = env.grasp_idx(env_ids_main, primitive_position, best_rotation_angle)
                # record
                reward_value = 1 if success else 0
                collector.executed_action_log.append(
                    [
                        1,  # grasp
                        primitive_position[0][0],
                        primitive_position[0][1],
                        primitive_position[0][2],
                        best_rotation_angle[0],
                        -1,
                        -1,
                    ]
                )
                collector.label_value_log.append(reward_value)
                collector.write_to_log("executed-action", collector.executed_action_log)
                collector.write_to_log("label-value", collector.label_value_log)
                iteration += 1
                if success:
                    num_action[2] += 1
                    break
                else:
                    continue
            # if new_q_value > GRASP_Q_TEST_THRESHOLD:
            #     action = mcts_helper.propose_grasp()
            #     if action is not None:
            #         best_rotation_angle = [action[3]]
            #         primitive_position = [[action[0], action[1], action[2]]]

            #         success = env.grasp_idx(env_ids_main, primitive_position, best_rotation_angle)
            #         # record
            #         reward_value = 1 if success else 0
            #         collector.executed_action_log.append(
            #             [
            #                 1,  # grasp
            #                 primitive_position[0][0],
            #                 primitive_position[0][1],
            #                 primitive_position[0][2],
            #                 best_rotation_angle[0],
            #                 -1,
            #                 -1,
            #             ]
            #         )
            #         collector.label_value_log.append(reward_value)
            #         collector.write_to_log("executed-action", collector.executed_action_log)
            #         collector.write_to_log("label-value", collector.label_value_log)
            #         iteration += 1
            #         if success:
            #             num_action[2] += 1
            #             break
            #         else:
            #             continue
            # Grasp <<<<<

            # Search >>>>>
            start_planning_time = time.time()
            success, object_states, _ = env.save_object_states(env_ids_main)
            assert success
            if new_q_value > GRASP_Q_PUSH_THRESHOLD and q_value <= GRASP_Q_GRASP_THRESHOLD:
                initial_state = PushState("root", object_states[0], q_value, 0, mcts_helper)
            else:
                initial_state = PushState("root", object_states[0], new_q_value, 0, mcts_helper)
            root = PushSearchNode(initial_state)
            mcts = MonteCarloTreeSearch(root, args.time_limit)
            # resolve the issue that mcts says we can grasp, but grasp net says no
            if new_q_value > GRASP_Q_PUSH_THRESHOLD and q_value <= GRASP_Q_GRASP_THRESHOLD:
                print("Changed to DQN for next push")
                PushState.grasp_method = "dqn"
                PushState.max_level = 1
            best_node = mcts.best_action(test)
            # print("best node:")
            # print(best_node.state.uid)
            # print(best_node.state.q_value)
            # print(sum(best_node.q) / len(best_node.q))
            # print(best_node.prev_move)
            node = best_node
            node_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            node_action = str(node.prev_move).split("_")
            cv2.arrowedLine(
                node_image,
                (int(node_action[1]), int(node_action[0])),
                (int(node_action[3]), int(node_action[2])),
                (255, 0, 255),
                2,
                tipLength=0.4,
            )
            node_push_result = mcts_helper.simulation_recorder[node.state.uid]
            node_push_image = cv2.cvtColor(node_push_result[1], cv2.COLOR_RGB2BGR)
            collector.save_predictions(iteration, node_image)
            collector.save_predictions(iteration + 10000, node_push_image)
            # collector.save_heightmaps(iteration + 10000, node_push_image, node_push_result[2])
            end_planning_time = time.time()
            planning_time += end_planning_time - start_planning_time
            # Search <<<<<

            # Push >>>>>
            num_action[1] += 1
            push_start = best_node.prev_move.pos0
            push_end = best_node.prev_move.pos1
            push_start = [
                [
                    push_start[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                    push_start[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                    0.01,
                ]
            ]
            push_end = [
                [
                    push_end[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                    push_end[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                    0.01,
                ]
            ]
            env.push_idx(env_ids_main, push_start, push_end)
            # record
            reward_value = 0
            collector.executed_action_log.append(
                [
                    0,
                    push_start[0][0],
                    push_start[0][1],
                    push_start[0][2],
                    push_end[0][0],
                    push_end[0][1],
                    push_end[0][2],
                ]  # push
            )
            collector.label_value_log.append(reward_value)
            collector.write_to_log("executed-action", collector.executed_action_log)
            collector.write_to_log("label-value", collector.label_value_log)
            iteration += 1
            # Push <<<<<

            # Plot
            if is_plot:
                collector.plot_mcts(env, root, iteration)

            # Save tree for training, BFS
            # best_action = best_node.prev_move
            # collector.record_image_idx.append(collector.idx)
            # collector.record_action.append(
            #     [best_action.pos0[1], best_action.pos0[0], best_action.pos1[1], best_action.pos1[0]]
            # )
            # label = 2
            # collector.record_label.append(label)
            if not test:
                collector.save_mcts_data(mcts_helper, env, root, best_node.prev_move, collector.idx)

            # clean up for memory
            del initial_state
            del mcts
            del root
            del best_node
            del push_start
            del push_end
            mcts_helper.reset()
            gc.collect()

            if num_action[0] > 15:
                print(Fore.RED + "Cannot solve it within 15 steps! End!")
                break

        end_time = time.time()
        collector.time_log.append(end_time - start_time)
        collector.planning_time_log.append(planning_time)
        collector.write_to_log("executed-time", collector.time_log)
        collector.write_to_log("planning-time", collector.planning_time_log)

        print(num_action)
        num_action_log[cases[repeat_idx]].append(num_action)

    print(num_action_log)
    total_case = 0
    total_action = 0
    for key in num_action_log:
        this_total_case = 0
        this_total_action = 0
        this_total_push = 0
        this_total_grasp = 0
        this_total_success = 0
        for trial in num_action_log[key]:
            this_total_case += 1
            this_total_action += trial[0]
            this_total_push += trial[1]
            this_total_grasp += trial[0] - trial[1]
            this_total_success += trial[2]
        if this_total_grasp == 0:
            average_grasp = 0
        else:
            average_grasp = this_total_success / this_total_grasp
        print(
            key,
            "this_case:",
            this_total_case,
            "this_action:",
            this_total_action,
            "this_push:",
            this_total_push,
            "this_grasp:",
            this_total_grasp,
            "average num",
            this_total_action / this_total_case,
            "average_grasp",
            average_grasp,
            "total_complete",
            this_total_success,
        )
        total_case += len(num_action_log[key])
        for re in num_action_log[key]:
            total_action += re[0]
    print(total_case, total_action, total_action / total_case)

    env.close()
    mcts_helper.close_pool()

