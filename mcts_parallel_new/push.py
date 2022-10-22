"""Class for MCTS."""


import torch

from constants import (
    MCTS_GRASP_SCALE,
    MCTS_PARALLEL_MAX_LEVEL,
    GRASP_Q_PUSH_THRESHOLD,
)
from colorama import Fore

from action_utils import (
    sample_actions_parallel,
    sample_actions_parallel_batch,
    sample_pre_defined_actions_parallel,
    sample_pre_defined_actions_parallel_batch,
)
from mcts_utils import MCTSHelper


class PushMove:
    """Represent a move from start to end pose"""

    def __init__(self, pos0, pos1):
        self.pos0 = pos0
        self.pos1 = pos1

    def __str__(self):
        return f"{self.pos0[0]}_{self.pos0[1]}_{self.pos1[0]}_{self.pos1[1]}"

    def __repr__(self):
        return f"start: {self.pos0} to: {self.pos1}"

    def __eq__(self, other):
        return self.pos0 == other.pos0 and self.pos1 == other.pos1

    def __hash__(self):
        return hash((self.pos0, self.pos1))


class PushState:
    """Use move_recorder and simulation_recorder from simulation.
    move_recorder, Key is uid: '(key of this)'.
    simulation_recorder, Key is the uid: '(key of this) + (level)' + (move).
    """

    max_level = MCTS_PARALLEL_MAX_LEVEL
    max_q = GRASP_Q_PUSH_THRESHOLD
    grasp_method = "classifier"

    def __init__(
        self, uid, object_states, q_value, level, mcts_helper: MCTSHelper, prev_move=None,
    ):
        self.uid = uid
        self.object_states = object_states
        self.q_value = q_value
        self.level = level
        self.mcts_helper = mcts_helper
        self.prev_move = prev_move

    @property
    def push_result(self):
        """Return the grasp q value"""
        result = self.q_value
        # sigmod in range [0, 1]
        result *= MCTS_GRASP_SCALE
        if self.q_value > PushState.max_q:
            result += 1
        return result

    def reset_max_props(self):
        PushState.max_level = MCTS_PARALLEL_MAX_LEVEL
        PushState.max_q = GRASP_Q_PUSH_THRESHOLD
        PushState.grasp_method = "classifier"

    @property
    def is_push_over(self):
        """Should stop the search"""
        # if reaches the last defined level or the object can be grasp
        if self.level >= PushState.max_level or self.q_value >= PushState.max_q:
            return True

        # if no legal actions
        if self.uid in self.mcts_helper.move_recorder:
            if len(self.mcts_helper.move_recorder[self.uid]) == 0:
                return True

        # if not over - no result
        return False

    def is_push_over_rollout(self, extra_level):
        """Should stop the search"""
        # if reaches the last defined level or the object can be grasp
        if PushState.max_level == MCTS_PARALLEL_MAX_LEVEL:
            if self.level >= max(PushState.max_level, self.level + extra_level) or self.q_value >= PushState.max_q:
                return True
        else:
            if self.level >= PushState.max_level or self.q_value >= PushState.max_q:
                return True

        # if no legal actions
        if self.uid in self.mcts_helper.move_recorder:
            if len(self.mcts_helper.move_recorder[self.uid]) == 0:
                return True

        # if not over - no result
        return False

    def _move_result(self, move, restore_state=True):
        """Return the result after a move"""

        key = f"{self.uid}.{self.level}-{move}"

        if key not in self.mcts_helper.simulation_recorder:
            action = [[move.pos0, move.pos1]]
            object_states, qualify = self.mcts_helper.simulate(
                self.mcts_helper.other_id, action, self.object_states, restore_state
            )

            if not qualify[0]:
                return None

            object_states = object_states[qualify]

            if PushState.grasp_method == "classifier":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    self.mcts_helper.other_id, color=True, depth=True, segm=False, focal_target=True
                )
                color_image = color_images[0]
                depth_image = depth_images[0]
                new_image_q = self.mcts_helper.grasp_eval(depth_image)
            elif PushState.grasp_method == "dqn":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    self.mcts_helper.other_id, color=True, depth=True, segm=False, focal_target=False
                )
                color_image = color_images[0]
                depth_image = depth_images[0]
                new_image_q, _, _ = self.mcts_helper.get_grasp_q(color_image, depth_image, post_checking=True)
            else:
                raise Exception

            # color_images, depth_images, _ = self.mcts_helper.env.render_camera(
            #     self.mcts_helper.other_id, color=True, depth=True, segm=False, focal_target=True
            # )
            # color_image = color_images[0]
            # depth_image = depth_images[0]
            # # segm_image = segm_images[0]

            # new_image_q = self.mcts_helper.grasp_eval(depth_image)
            self.mcts_helper.simulation_recorder[key] = object_states, color_image, depth_image, new_image_q
        else:
            object_states, _, _, new_image_q = self.mcts_helper.simulation_recorder[key]

        return object_states, new_image_q

    def move(self, move, restore_state=True):
        result = self._move_result(move, restore_state)
        if result is None:
            return None
        object_states, new_image_q = result
        move_in_image = ((move.pos0[1], move.pos0[0]), (move.pos1[1], move.pos1[0]))
        new_state = PushState(
            f"{self.uid}.{self.level}-{move}",
            object_states,
            new_image_q,
            self.level + 1,
            self.mcts_helper,
            prev_move=move_in_image,
        )
        if new_state.level < PushState.max_level and new_state.q_value >= PushState.max_q:
            print(Fore.GREEN + f"Found a solution during move, {new_state.uid} at level {new_state.level}")
            PushState.max_level = new_state.level
        return new_state

    def get_actions(self, color_image=None):
        key = self.uid
        if key not in self.mcts_helper.move_recorder:
            # Retrieve information
            if color_image is None:
                _, color_image, _, _ = self.mcts_helper.simulation_recorder[key]

            # Sample actions
            states = self.object_states.cpu().numpy()[2:]
            actions = sample_pre_defined_actions_parallel(
                color_image, self.mcts_helper.env.defined_actions, states, self.mcts_helper.pool, plot=False
            )
            moves = []
            for action in actions:
                moves.append(PushMove(action[0], action[1]))
            self.mcts_helper.move_recorder[key] = moves
        else:
            moves = self.mcts_helper.move_recorder[key]

        return moves

    def remove_action(self, move):
        key = self.uid
        if key in self.mcts_helper.move_recorder:
            moves = self.mcts_helper.move_recorder[key]
            if move in moves:
                moves.remove(move)

    def first_expand(self, color_image=None):
        """first level"""

        # Retrieve information
        if color_image is None:
            # use the second (anyone except the first) env to sample actions
            success, _ = self.mcts_helper.env.restore_object_states(self.mcts_helper.other_id, self.object_states)
            assert success
            # assert success
            color_images, _, _ = self.mcts_helper.env.render_camera(
                self.mcts_helper.other_id, color=True, depth=False, segm=False
            )
            color_image = color_images[0]

        key = self.uid

        # Sample actions
        states = self.object_states.cpu().numpy()[2:]
        actions = sample_pre_defined_actions_parallel(
            color_image, self.mcts_helper.env.defined_actions, states, self.mcts_helper.pool, plot=False
        )
        print(f"Process first level with {len(actions)} actions")

        # Simulate actions
        if len(actions) > 0:
            env_ids = self.mcts_helper.all_other_id[: len(actions)]
            # env_ids = torch.arange(1, len(actions) + 1, device=self.mcts_helper.device)
            object_states, qualify = self.mcts_helper.simulate(env_ids, actions, self.object_states)
            env_ids = env_ids[qualify]
            actions = torch.tensor(actions)[qualify].int().tolist()
            object_states = object_states[qualify]
            assert len(env_ids) == len(object_states)

        # Record actions
        first_moves = []
        move_results = []
        if len(actions) > 0:
            # prepare data for buffer
            if PushState.grasp_method == "classifier":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    env_ids, color=True, depth=True, segm=False, focal_target=True
                )
                grasp_values = self.mcts_helper.grasp_eval_batch(depth_images)
            elif PushState.grasp_method == "dqn":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    env_ids, color=True, depth=True, segm=False, focal_target=False
                )
                grasp_values = self.mcts_helper.get_grasp_q_batch(color_images, depth_images, post_checking=True)
                print(grasp_values)
            else:
                raise NotImplementedError
            group = zip(actions, object_states, color_images, depth_images, grasp_values)

            # save buffer
            for data in group:
                action, object_state, color_image, depth_image, grasp_value = data
                first_moves.append(PushMove(action[0], action[1]))
                sim_key = f"{self.uid}.{self.level}-{first_moves[-1]}"
                result = (object_state, color_image, depth_image, grasp_value)
                move_results.append(result)
                self.mcts_helper.simulation_recorder[sim_key] = result
        else:
            raise Exception("No possible actions!!!!!")
        self.mcts_helper.move_recorder[key] = first_moves

        # Create new states and nodes
        first_states = []
        for idx, result in enumerate(move_results):
            move = first_moves[idx]
            object_states, _, _, new_image_q = result
            move_in_image = ((move.pos0[1], move.pos0[0]), (move.pos1[1], move.pos1[0]))
            new_state = PushState(
                f"{self.uid}.{self.level}-{move}",
                object_states,
                new_image_q,
                self.level + 1,
                self.mcts_helper,
                prev_move=move_in_image,
            )
            first_states.append(new_state)

        return first_states, first_moves, grasp_values

    def batch_get_actions(self, nodes_states):
        """Assume the depth of any node is not at the last level"""

        # # Retrieve images
        # color_images = []
        # mask_images = []
        # all_actions = []
        # for state in nodes_states:
        #     key = state.uid
        #     if key not in self.mcts_helper.move_recorder:
        #         _, color_image, depth_image, segm_image, _ = self.mcts_helper.simulation_recorder[key]
        #         color_images.append(color_image)
        #         mask_images.append(segm_image)
        #         all_actions.append(None)
        #     else:
        #         all_actions.append(self.mcts_helper.move_recorder[key])

        # partial_actions = sample_actions_parallel_batch(color_images, mask_images, self.mcts_helper.pool)

        # Retrieve images
        partial_inputs = []
        all_actions = []
        for state in nodes_states:
            key = state.uid
            if key not in self.mcts_helper.move_recorder:
                object_state, color_image, _, _ = self.mcts_helper.simulation_recorder[key]
                object_state = object_state.cpu().numpy()[2:]
                partial_inputs.append([color_image, self.mcts_helper.env.defined_actions, object_state])
                all_actions.append(None)
            else:
                moves = self.mcts_helper.move_recorder[key]
                actions_list = []
                for move in moves:
                    actions_list.append([move.pos0, move.pos1])
                all_actions.append(actions_list)
                # all_actions.append(self.mcts_helper.move_recorder[key])

        partial_actions = sample_pre_defined_actions_parallel_batch(partial_inputs, self.mcts_helper.pool)

        partial_idx = 0
        for i in range(len(all_actions)):
            if all_actions[i] is None:
                actions = partial_actions[partial_idx]
                moves = []
                for action in actions:
                    moves.append(PushMove(action[0], action[1]))
                self.mcts_helper.move_recorder[nodes_states[i].uid] = moves
                all_actions[i] = actions
                partial_idx += 1

        return all_actions

        # # Sample actions
        # all_actions = []
        # action_idx_start = []
        # action_idx_end = []
        # group_id = []
        # num_actions = 0
        # for i in range(len(color_images)):
        #     key = node_states[i].uid
        #     if key not in self.mcts_helper.move_recorder:
        #         actions = sample_actions_parallel(color_images[i], mask_images[i], self.mcts_helper.pool, plot=False)
        #     else:
        #         actions = self.mcts_helper.move_recorder[key]
        #     all_actions.extend(actions)
        #     action_idx_start.append(num_actions)
        #     num_actions += len(actions)
        #     action_idx_end.append(num_actions)
        #     group_id.extend([i] * len(actions))

        # all_states = torch.zeros(
        #     (len(all_actions), self.object_states.shape[1], self.object_states.shape[2]),
        #     device=self.mcts_helper.device,
        # )
        # for i in range(len(action_idx_start)):
        #     i_s = action_idx_start[i]
        #     i_e = action_idx_end[i]
        #     all_states[i_s:i_e, ...] = node_states[i].object_states

        # return all_states, all_actions, group_id

    def batch_move(self, all_node_states, all_object_states, all_actions):

        # Simulate actions
        env_ids = self.mcts_helper.all_other_id[: len(all_actions)]
        # env_ids = torch.arange(1, len(all_actions) + 1, device=self.mcts_helper.device)
        object_states, qualify = self.mcts_helper.simulate(env_ids, all_actions, all_object_states)
        env_ids = env_ids[qualify]
        node_states = [all_node_states[i] for i in range(len(qualify)) if qualify[i]]
        actions = [all_actions[i] for i in range(len(qualify)) if qualify[i]]
        object_states = object_states[qualify]
        assert len(env_ids) == len(object_states)

        # Record actions and create states
        recorded_states = []
        recorded_actions = []
        if len(actions) > 0:
            # prepare data for buffer
            if PushState.grasp_method == "classifier":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    env_ids, color=True, depth=True, segm=False, focal_target=True
                )
                grasp_values = self.mcts_helper.grasp_eval_batch(depth_images)
            elif PushState.grasp_method == "dqn":
                color_images, depth_images, _ = self.mcts_helper.env.render_camera(
                    env_ids, color=True, depth=True, segm=False, focal_target=False
                )
                grasp_values = self.mcts_helper.get_grasp_q_batch(color_images, depth_images, post_checking=True)
            else:
                raise NotImplementedError
            # color_images, depth_images, _ = self.mcts_helper.env.render_camera(
            #     env_ids, color=True, depth=True, segm=False, focal_target=True
            # )
            # grasp_values = self.mcts_helper.grasp_eval_batch(depth_images)
            group = zip(node_states, actions, object_states, color_images, depth_images, grasp_values)

            # save buffer
            for data in group:
                node_state, action, object_state, color_image, depth_image, grasp_value = data
                move = PushMove(action[0], action[1])
                result = (object_state, color_image, depth_image, grasp_value)
                sim_key = f"{node_state.uid}.{node_state.level}-{move}"

                # record action
                self.mcts_helper.simulation_recorder[sim_key] = result
                # if node_state.uid not in self.mcts_helper.move_recorder:
                #     self.mcts_helper.move_recorder[node_state.uid] = [move]
                # else:
                #     self.mcts_helper.move_recorder[node_state.uid].append(move)

                # create states
                move_in_image = ((move.pos0[1], move.pos0[0]), (move.pos1[1], move.pos1[0]))
                new_state = PushState(
                    sim_key, object_state, grasp_value, node_state.level + 1, self.mcts_helper, prev_move=move_in_image,
                )

                # prune the tree search
                if new_state.level < PushState.max_level and new_state.q_value >= PushState.max_q:
                    print(
                        Fore.GREEN + f"Found a solution during batch move, {new_state.uid} at level {new_state.level}"
                    )
                    PushState.max_level = new_state.level

                recorded_states.append(new_state)
                recorded_actions.append(move)

        return recorded_states, recorded_actions, qualify

