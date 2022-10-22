"""Node for MCTS"""

import numpy as np
from constants import MCTS_DISCOUNT, MCTS_ROLLOUT_EXTRA_LEVEL, MCTS_UCT_RATIO
from mcts_ori.push import PushState


class PushSearchNode:
    """MCTS search node for push prediction."""

    def __init__(self, state: PushState = None, prev_move=None, parent=None):
        self.state = state
        self.prev_move = prev_move
        self.parent = parent
        self.children = []
        self._number_of_visits = 0
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
    def has_children(self):
        return len(self.children) > 0

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def is_terminal_node(self):
        return self.state.is_push_over or (self.is_fully_expanded and not self.has_children)

    def expand(self):
        expanded = False
        child_node = self

        while len(self.untried_actions) > 0:
            action = self.untried_actions.pop()
            result = self.state.move(action)
            if result is None:
                self.state.remove_action(action)
            else:
                next_state = result
                child_node = PushSearchNode(next_state, action, parent=self)
                self.children.append(child_node)
                expanded = True
                break

        return expanded, child_node

    def rollout(self):
        current_rollout_state = self.state
        discount_accum = 1
        rollout_left_level = MCTS_ROLLOUT_EXTRA_LEVEL
        results = [current_rollout_state.push_result]
        restore_state = True
        color_image = None
        segm_image = None
        while not current_rollout_state.is_push_over_rollout(rollout_left_level):
            rollout_left_level -= 1

            possible_moves = current_rollout_state.get_actions(color_image, segm_image)
            if len(possible_moves) == 0:
                break
            action = self.rollout_policy(possible_moves)
            new_rollout_state = current_rollout_state.move(action, restore_state)

            if new_rollout_state is None:
                if current_rollout_state == self.state:
                    self.untried_actions.remove(action)
                current_rollout_state.remove_action(action)
                color_image = None
                segm_image = None
                restore_state = True
            else:
                discount_accum *= MCTS_DISCOUNT
                current_rollout_state = new_rollout_state
                results.append(current_rollout_state.push_result * discount_accum)
                (_, color_image, _, segm_image, _,) = current_rollout_state.mcts_helper.simulation_recorder[
                    current_rollout_state.uid
                ]
                restore_state = False
            print(
                current_rollout_state.uid,
                current_rollout_state.level,
                current_rollout_state.q_value,
                rollout_left_level,
            )

        return np.max(results)

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results.append(result)
        # self._results.sort(reverse=True)
        # self._results = self._results[:MCTS_TOP]
        result = max(result, self.state.push_result)  # TODO: need this ?
        if self.parent:
            self.parent.backpropagate(result * MCTS_DISCOUNT)

    def best_child(self, c_param=MCTS_UCT_RATIO):
        # choices_weights = [sum(c.q) / len(c.q) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        choices_weights = [sum(c.q) / c.n + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    # def best_child_top(self):
    #     choices_weights = [max(c.q) for c in self.children]
    #     return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[self.state.mcts_helper.np_rng.integers(len(possible_moves))]

