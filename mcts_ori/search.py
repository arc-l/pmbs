import time
from colorama import Fore

from mcts_ori.push import PushState


class MonteCarloTreeSearch(object):
    def __init__(self, node, time_limit):
        self.root = node
        self.time_limit = time_limit
        self.root.state.reset_max_props()  # refresh variables

    def best_action(self, eval=False):
        start_time = time.time()

        stop_level = 1
        early_stop_check_nodes = [self.root]
        active_nodes = [self.root]
        solution_nodes = []

        itr = 0
        found_solution = False
        curr_time = time.time()
        duration = curr_time - start_time
        while duration < self.time_limit and not found_solution and len(active_nodes) > 0:
            self.clean_terminal_or_fully_expanded(active_nodes, True)

            print(Fore.GREEN + "select + expand")
            s = time.time()
            child_node = self._tree_policy()
            e = time.time()
            print(f"select level {child_node.state.level} {e - s:.2f}")

            if child_node.state.uid not in child_node.state.mcts_helper.move_recorder and not child_node.state.is_push_over:
                active_nodes.append(child_node)
            elif not child_node.is_terminal_node and not child_node.is_fully_expanded:
                active_nodes.append(child_node)

            if eval:
                if len(early_stop_check_nodes) > 0:
                    self.clean_terminal_or_fully_expanded(early_stop_check_nodes, True)
                if len(early_stop_check_nodes) == 0:
                    stop_level += 1
                    early_stop_check_nodes = self.add_all_at_nodes_at_level(stop_level - 1)
                    print(f"Update stop level {stop_level}")
                if PushState.max_level < stop_level:
                    PushState.max_level = stop_level

                # stop early if a solutin has been found such that all states at the same level have been explored
                if child_node.state.q_value >= PushState.max_q:
                    solution_nodes.append(child_node)
                for node in solution_nodes:
                    if node.state.level <= stop_level:
                        found_solution = True
                        print(f"Early stop: found solution in level {node.state.level}")
                        print(f"Node {node.state.uid}, grasp reward: {node.state.push_result}")
                        break

            # # Use DQN for next two levels after root, assume the classifier is the default
            # if PushState.grasp_method == "dqn":
            #     if PushState.max_level > 1:
            #         PushState.max_level = 1
            #     if stop_level > 1:
            #         stop_level = 1

            print(Fore.GREEN + "rollout + backup")
            s = time.time()
            if not found_solution:
                reward = child_node.rollout()
            else:
                reward = child_node.state.push_result
            child_node.backpropagate(reward)
            e = time.time()
            print(f"backup reward {reward} {e - s:.2f}")

            curr_time = time.time()
            duration = curr_time - start_time
            print(Fore.GREEN + f"Iteration: {itr}; Time: {duration:.2f} / {self.time_limit} s")
            itr += 1
        # to select best child go for exploitation only
        # return self.root.best_child_top()
        return self.root.best_child(c_param=0)

    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node or current_node == self.root:
            if current_node == self.root and current_node.is_terminal_node:
                break
            if not current_node.is_fully_expanded:
                expanded, node = current_node.expand()
                if expanded:
                    return node
                else:
                    current_node = self.root
            else:
                if current_node.has_children:
                    current_node = current_node.best_child()
        # if current_node.is_terminal_node:
        #     current_node._number_of_visits += 10  # encourage to explore other nodes
        return current_node

    def clean_terminal_or_fully_expanded(self, nodes: list, end_early: bool):
        to_be_delete = []
        for node in nodes:
            if node.state.uid not in node.state.mcts_helper.move_recorder and not node.state.is_push_over:
                if end_early:
                    break
            # if node.state.uid not in node.state.mcts_helper.move_recorder:
            #     if end_early:
            #         break
            elif node.is_terminal_node or node.is_fully_expanded:
                to_be_delete.append(node)
            else:
                if end_early:
                    break
        for node in to_be_delete:
            nodes.remove(node)

    def add_all_at_nodes_at_level(self, level: int):
        """"BFS"""
        nodes = []
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop(0)
            if node.state.level == level:
                nodes.append(node)
            elif node.state.level > level:
                break
            else:
                queue.extend(node.children)
        return nodes
