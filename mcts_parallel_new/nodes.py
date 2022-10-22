"""Node for MCTS"""

import time
import numpy as np
import torch
from constants import MCTS_PARALLEL_DISCOUNT, MCTS_PARALLEL_UCT_RATIO, MCTS_ROLLOUT_EXTRA_LEVEL
from mcts_parallel_new.push import PushMove, PushState


class PushSearchNode:
    """MCTS search node for push prediction."""

    def __init__(self, state: PushState = None, prev_move=None, parent=None):
        self.state = state
        self.prev_move = prev_move
        self.parent = parent
        self.children = []
        self._number_of_visits = 0
        self._number_of_virtual_visits = 0
        # self._virtual_selected = False
        self._results = []
        self._untried_actions = None

    def __eq__(self, __o: object) -> bool:
        return self.state.uid == __o.state.uid

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_actions().copy()
        return self._untried_actions

    @property
    def q(self):
        return self._results

    @property
    def n(self):
        return self._number_of_visits

    @property
    def vn(self):
        return self._number_of_virtual_visits

    @property
    def has_children(self):
        return len(self.children) > 0

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def is_terminal_node(self):
        return self.state.is_push_over or (self.is_fully_expanded and not self.has_children)

    def first_expand(self):
        """"Assume the expand ends at level one, and their grasp_q will be used as reward
        All pre-expanded nodes have been just visited once
        """
        # only start from the root
        assert self.parent is None

        best_state = None
        stop_level = 2

        first_states, first_moves, grasp_values = self.state.first_expand()

        # the depth 1
        self._untried_actions = []
        for i, state in enumerate(first_states):
            child = PushSearchNode(state, first_moves[i], parent=self)
            child._number_of_visits = 1
            child._results.append(state.push_result)
            self.children.append(child)
            self._number_of_visits += 1

        max_q_so_far = np.max(grasp_values)
        best_state = first_states[np.argmax(grasp_values)]

        if max_q_so_far >= PushState.max_q:
            stop_level = 1
        print(f"max q so far {max_q_so_far}, first")

        return best_state, stop_level

    def batch_expand(self, candidate_nodes, candidate_actions):
        """Assume the depth of any node is not at the last level"""

        assert len(candidate_nodes) == len(candidate_actions)

        # Prepare data
        all_object_states = []
        all_actions = []
        actions_to_try = []
        parent_nodes = []
        parent_node_states = []
        for node, action in zip(candidate_nodes, candidate_actions):
            all_object_states.append(node.state.object_states)
            all_actions.append([action.pos0, action.pos1])
            actions_to_try.append(action)
            parent_nodes.append(node)
            parent_node_states.append(node.state)
        all_object_states = torch.stack(all_object_states)

        # Simulate
        recorded_states, recorded_actions, qualify = self.state.batch_move(
            parent_node_states, all_object_states, all_actions
        )
        recorded_parent_nodes = []
        for node, state, action, qual in zip(parent_nodes, parent_node_states, actions_to_try, qualify):
            if not qual:
                state.remove_action(action)
            else:
                recorded_parent_nodes.append(node)

        # Expand
        unique_parent_nodes = []
        child_nodes = []
        for state, action, parent in zip(recorded_states, recorded_actions, recorded_parent_nodes):
            child = PushSearchNode(state, action, parent=parent)
            parent.children.append(child)
            child_nodes.append(child)
            if parent not in unique_parent_nodes:
                unique_parent_nodes.append(parent)

        return unique_parent_nodes, child_nodes, recorded_states

    def batch_rollout(self, nodes_states, total_num_envs, time_limit):
        """For each node, we will try a rollout all the way to the end
        If number of node of state is smaller than num_envs, we will try to randomly assign free resource to states
        """

        start_time = time.time()

        rollout_left_level = MCTS_ROLLOUT_EXTRA_LEVEL

        # If it is a terminal node already, we should not do rollout with it
        rewards = []
        valid_idx = []
        for i, node_state in enumerate(nodes_states):
            if not node_state.is_push_over_rollout(rollout_left_level):
                valid_idx.append(i)
            rewards.append(node_state.push_result)

        min_num_state = total_num_envs // 10  # it can be 0, but lots of resource will be wasted
        discount_accum = 1
        rollout_nodes_states = [nodes_states[i] for i in valid_idx]

        curr_time = time.time()
        duration = curr_time - start_time

        while duration < time_limit and len(rollout_nodes_states) > min_num_state:
            rollout_left_level -= 1

            # Assign jobs
            all_actions = self.state.batch_get_actions(rollout_nodes_states)
            num_states = len(rollout_nodes_states)
            envs_per_state = total_num_envs // num_states
            print(f"rollout envs_per_state {envs_per_state} with dicount {discount_accum}")
            free_envs = total_num_envs - envs_per_state * num_states
            num_envs = [envs_per_state] * num_states
            if free_envs > 0:
                for i in range(num_states):
                    if len(all_actions[i]) <= envs_per_state:
                        num_envs[i] = len(all_actions[i])
                        free_envs += envs_per_state - len(all_actions[i])
                    else:
                        num_left_action = len(all_actions[i]) - envs_per_state
                        assign_num = min(free_envs, num_left_action)
                        num_envs[i] += assign_num
                        free_envs -= assign_num
                        if free_envs == 0:
                            break
            assert free_envs >= 0

            # Sample actions randomly
            sampled_actions = []
            parent_node_states = []
            all_object_states = []
            parent_idx = []
            for num, actions, state, vi in zip(num_envs, all_actions, rollout_nodes_states, valid_idx):
                indices = self.state.mcts_helper.np_rng.choice(list(range(len(actions))), num, replace=False)
                for i in indices:
                    sampled_actions.append(actions[i])
                parent_node_states.extend([state] * num)
                all_object_states.extend([state.object_states] * num)
                parent_idx.extend([vi] * num)
            all_object_states = torch.stack(all_object_states)

            # Simulate
            recorded_states, _, qualify = self.state.batch_move(parent_node_states, all_object_states, sampled_actions)
            for state, action, qual in zip(parent_node_states, sampled_actions, qualify):
                if not qual:
                    state.remove_action(PushMove(action[0], action[1]))

            # Update reward and prepare for next iteration
            new_valid_idx = []
            new_rollout_nodes_states = []
            recorded_idx = 0
            for qual, pi in zip(qualify, parent_idx):
                if qual:
                    new_state = recorded_states[recorded_idx]
                    r = new_state.push_result * discount_accum
                    if r > rewards[pi]:
                        rewards[pi] = r

                    if not new_state.is_push_over_rollout(rollout_left_level):
                        new_valid_idx.append(pi)
                        new_rollout_nodes_states.append(new_state)

                    recorded_idx += 1
            assert recorded_idx == len(recorded_states)
            discount_accum *= MCTS_PARALLEL_DISCOUNT
            valid_idx = new_valid_idx
            rollout_nodes_states = new_rollout_nodes_states

            curr_time = time.time()
            duration = curr_time - start_time

        # # Assign jobs
        # all_actions = self.state.batch_get_actions(nodes_states)
        # num_actions = len(all_actions)
        # envs_per_state = num_envs // num_actions
        # free_envs = num_envs - envs_per_state * num_actions
        # num_envs = [envs_per_state] * num_actions
        # if free_envs > 0:
        #     for i in range(num_actions):
        #         if len(all_actions[i]) <= envs_per_state:
        #             num_envs[i] = len(all_actions[i])
        #             free_envs += envs_per_state - len(all_actions[i])
        #         else:
        #             num_left_action = len(all_actions[i]) - envs_per_state
        #             assign_num = min(free_envs, num_left_action)
        #             num_envs[i] += assign_num
        #             free_envs -= assign_num
        #             if free_envs == 0:
        #                 break
        # assert free_envs >= 0

        # # random sample
        # sampled_actions = []
        # parent_nodes = []
        # parent_node_states = []
        # all_object_states = []
        # for num, actions, node, state in zip(num_envs, all_actions, nodes, nodes_states):
        #     # TODO: fix ValueError: Cannot take a larger sample than population when 'replace=False'
        #     indices = self.state.mcts_helper.np_rng.choice(list(range(len(actions))), num, replace=False)
        #     for i in indices:
        #         sampled_actions.append(actions[i])
        #     parent_nodes.extend([node] * num)
        #     parent_node_states.extend([state] * num)
        #     all_object_states.extend([state.object_states] * num)
        # all_object_states = torch.stack(all_object_states)

        # # Simulate
        # recorded_states, _, qualify = self.state.batch_move(parent_node_states, all_object_states, sampled_actions)
        # for state, action, qual in zip(parent_node_states, sampled_actions, qualify):
        #     if not qual:
        #         state.remove_action(PushMove(action[0], action[1]))

        # rewards = []
        # parent_idx = 0
        # for i in range(len(qualify)):
        #     if qualify[i]:
        #         rewards.append(recorded_states[parent_idx].push_result * MCTS_PARALLEL_DISCOUNT)
        #         parent_idx += 1
        #     else:
        #         rewards.append(parent_node_states[i].push_result)

        # assert parent_idx == len(recorded_states)

        return rewards

    # def expand(self):
    #     expanded = False
    #     child_node = self

    #     while len(self.untried_actions) > 0:
    #         action = self.untried_actions.pop()
    #         result = self.state.move(action)
    #         if result is None:
    #             self.state.remove_action(action)
    #         else:
    #             next_state = result
    #             child_node = PushSearchNode(next_state, action, parent=self)
    #             self.children.append(child_node)
    #             expanded = True
    #             break

    #     return expanded, child_node

    # def rollout(self):
    #     current_rollout_state = self.state
    #     discount_accum = 1
    #     results = [current_rollout_state.push_result]
    #     restore_state = True
    #     color_image = None
    #     mask_image = None
    #     while not current_rollout_state.is_push_over():
    #         possible_moves = current_rollout_state.get_actions(color_image, mask_image)
    #         if len(possible_moves) == 0:
    #             break
    #         action = self.rollout_policy(possible_moves)
    #         new_rollout_state = current_rollout_state.move(action, restore_state)

    #         if new_rollout_state is None:
    #             if current_rollout_state == self.state:
    #                 self.untried_actions.remove(action)
    #             current_rollout_state.remove_action(action)
    #             color_image = None
    #             mask_image = None
    #             restore_state = True
    #         else:
    #             discount_accum *= MCTS_PARALLEL_DISCOUNT
    #             current_rollout_state = new_rollout_state
    #             results.append(current_rollout_state.push_result * discount_accum)
    #             (_, color_image, mask_image, _,) = current_rollout_state.mcts_helper.simulation_recorder[
    #                 current_rollout_state.uid
    #             ]
    #             restore_state = False

    #     return np.max(results)

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results.append(result)
        # self._results.sort(reverse=True)
        # self._results = self._results[:MCTS_PARALLEL_TOP]  # TODO: change to a tree PQ
        result = max(result, self.state.push_result)  # TODO: need this ?
        if self.parent:
            self.parent.backpropagate(result * MCTS_PARALLEL_DISCOUNT)

    # @property
    # def is_virtual_selected(self):
    #     return self._virtual_selected

    # def mark_virtual_selected(self):
    #     self._virtual_selected = True

    def backpropagate_virtual(self, value=1, discount=1):  # TODO: tune this
        self._number_of_virtual_visits += value
        if self.parent:
            self.parent.backpropagate_virtual(value * discount)

    def clean_virtual_visits(self):
        self._number_of_virtual_visits = 0
        # self._virtual_selected = False
        if self.parent:
            self.parent.clean_virtual_visits()

    def best_child(self, c_param=MCTS_PARALLEL_UCT_RATIO, virtual_cost=0.1):  # TODO: tune this
        # choices_weights = [
        #     sum(c.q) / len(c.q) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) - virtual_cost * c.vn
        #     for c in self.children
        # ]
        choices_weights = [
            sum(c.q) / (c.n + c.vn) + c_param * np.sqrt((2 * np.log(self.n + self.vn) / (c.n + c.vn))) for c in self.children
        ]
        # choices_weights = [
        #     0 * sum(c.q) / (c.n + c.vn) + c_param * np.sqrt((2 * np.log(self.n + self.vn) / (c.n + c.vn))) for c in self.children
        # ]
        # choices_weights = [
        #     sum(c.q) / len(c.q) + c_param * np.sqrt((2 * np.log(self.n + self.vn) / (c.n + c.vn))) for c in self.children
        # ]
        return self.children[np.argmax(choices_weights)]

    # def best_child_top(self):
    #     choices_weights = [max(c.q) for c in self.children]
    #     return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[self.state.mcts_helper.np_rng.integers(len(possible_moves))]

