"""Class for MCTS."""

from colorama import Fore
from constants import (
    MCTS_GRASP_SCALE,
    MCTS_MAX_LEVEL,
    GRASP_Q_PUSH_THRESHOLD,
)

from action_utils import sample_actions, predict_action_q, sample_actions_parallel, sample_pre_defined_actions_parallel
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

    max_level = MCTS_MAX_LEVEL
    max_q = GRASP_Q_PUSH_THRESHOLD
    grasp_method = "classifier"

    # TODO: how to get a good max_q, which could be used to decide an object is graspable
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
        PushState.max_level = MCTS_MAX_LEVEL
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
        if PushState.max_level == MCTS_MAX_LEVEL:
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

            object_states = object_states[qualify][0]

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
                raise NotImplementedError

            # color_images, depth_images, _ = self.mcts_helper.env.render_camera(
            #     self.mcts_helper.other_id, color=True, depth=True, segm=False, focal_target=True
            # )
            # color_image = color_images[0]
            # depth_image = depth_images[0]
            # # segm_image = segm_images[0]

            # new_image_q = self.mcts_helper.grasp_eval(depth_image)
            # # new_image_q, _, _ = self.mcts_helper.get_grasp_q(color_image, depth_image, post_checking=True)
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
            # if color_image is None:
            #     # use the second (anyone except the first) env to sample actions
            #     success, _ = self.mcts_helper.env.restore_object_states(self.mcts_helper.other_id, self.object_states)
            #     assert success
            #     color_images, _, segm_images = self.mcts_helper.env.render_camera(
            #         self.mcts_helper.other_id, color=True, depth=False, segm=True
            #     )
            #     color_image = color_images[0]
            #     mask_image = segm_images[0]
            if color_image is None:
                if key not in self.mcts_helper.simulation_recorder:
                    assert self.level == 0
                    # use the second (anyone except the first) env to sample actions
                    success, _ = self.mcts_helper.env.restore_object_states(
                        self.mcts_helper.other_id, self.object_states
                    )
                    assert success
                    color_images, _, _ = self.mcts_helper.env.render_camera(
                        self.mcts_helper.other_id, color=True, depth=False, segm=False
                    )
                    color_image = color_images[0]
                else:
                    _, color_image, _, _ = self.mcts_helper.simulation_recorder[key]

            # Sample actions
            states = self.object_states.cpu().numpy()[2:]
            actions = sample_pre_defined_actions_parallel(
                color_image, self.mcts_helper.env.defined_actions, states, self.mcts_helper.pool, plot=False
            )
            # actions = sample_actions_parallel(color_image, mask_image, self.mcts_helper.pool, plot=False)
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

