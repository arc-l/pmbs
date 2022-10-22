import time
from mcts_parallel_new.nodes import PushSearchNode
from mcts_parallel_new.push import PushState
from colorama import Fore


class MonteCarloTreeSearch(object):
    def __init__(self, node: PushSearchNode, num_envs, time_limit):
        self.num_envs = num_envs
        self.time_limit = time_limit
        self.root = node
        self.root.state.reset_max_props()  # refresh variables

    def best_action_parallel(self, eval=False):
        """The batch expand could have waste, as there could be not enough nodes to expand
        The batch rollout will try to avoid waste, see batch_rollout for more details"""
        start_time = time.time()
        best_state, stop_level = self.root.first_expand()
        # Use DQN for only first levelsafter root
        if PushState.grasp_method == "dqn":
            stop_level = 1
            PushState.max_level = 1

        if stop_level == 2:
            solution_nodes = []
            early_stop_check_nodes = self.add_all_at_nodes_at_level(stop_level - 1)
            active_nodes = self.root.children.copy()

            itr = 0
            found_solution = False
            curr_time = time.time()
            duration = curr_time - start_time
            print(f"First Level Time: {duration:.2f} / {self.time_limit} s")
            while duration < self.time_limit and not found_solution and len(active_nodes) > 0:
                # Selection
                print(Fore.GREEN + "select")
                s = time.time()
                self.clean_terminal_or_fully_expanded(active_nodes, True)  # update active nodes
                candidate_nodes, candidate_actions, nodes_to_be_clean = self._parallel_tree_policy(active_nodes)
                if len(candidate_nodes) == 0:  # couldn't find a good solution
                    print(Fore.RED + "could not get candidate_nodes")
                    break
                for node in nodes_to_be_clean:
                    node.clean_virtual_visits()
                e = time.time()
                print("selected ", len(candidate_nodes), f"{e - s:.2f}")

                # Expansion
                print(Fore.GREEN + "expand")
                s = time.time()
                unique_parent_nodes, child_nodes, child_states = self.root.batch_expand(
                    candidate_nodes, candidate_actions
                )
                active_nodes.extend(child_nodes)
                e = time.time()
                print("parent", len(unique_parent_nodes), "child", len(child_nodes), f"{e - s:.2f}")

                # Check for early stop
                if eval:
                    s = time.time()
                    if len(early_stop_check_nodes) > 0:
                        self.clean_terminal_or_fully_expanded(early_stop_check_nodes, True)
                    if len(early_stop_check_nodes) == 0:
                        stop_level += 1
                        print(f"Update stop level {stop_level}")
                        early_stop_check_nodes = self.add_all_at_nodes_at_level(stop_level - 1)
                        print(f"early_stop_check_nodes have {len(early_stop_check_nodes)}")
                    if PushState.max_level < stop_level:
                        PushState.max_level = stop_level

                    # stop early if a solutin has been found such that all states at the same level have been explored
                    for node in child_nodes:
                        if node.state.q_value >= PushState.max_q and node not in solution_nodes:
                            solution_nodes.append(node)
                    for node in solution_nodes:
                        if node.state.level <= stop_level:
                            print(Fore.GREEN + f"Found solution early stop {node.state.level}")
                            print(f"Node {node.state.uid}, grasp reward: {node.state.push_result}")
                            found_solution = True
                            break

                    e = time.time()
                    print("early stop checking", f"{e - s:.2f}")

                curr_time = time.time()
                duration = curr_time - start_time
                if duration >= self.time_limit:
                    print('time is up!')
                    break

                # Rollout
                if not found_solution:
                    print(Fore.GREEN + "rollout")
                    s = time.time()
                    rewards = self.root.batch_rollout(child_states, self.num_envs, self.time_limit - duration)
                    e = time.time()
                    print(f"num of reward {len(rewards)}", f"{e - s:.2f}")

                # Backup
                print(Fore.GREEN + "backup")
                s = time.time()
                if found_solution:
                    for child_node in child_nodes:
                        child_node.backpropagate(child_node.state.push_result)
                else:
                    assert len(rewards) == len(child_nodes)
                    for reward, node in zip(rewards, child_nodes):
                        node.backpropagate(reward)
                e = time.time()
                print(f"backup", f"{e - s:.2f}")

                # if rewards is not None:
                #     for reward, node in zip(rewards, rollout_nodes):
                #         node.backpropagate(reward)
                #     for pi in invalid_idx:
                #         child_nodes[pi].backpropagate(child_nodes[pi].state.push_result)
                # else:
                #     for child_node in child_nodes:
                #         child_node.backpropagate(child_node.state.push_result)
                # e = time.time()
                # print(len(child_nodes), e - s)

                # # Rollout, just do one level rollout, we assume max depth is 3
                # rewards = None
                # if not found_solution:
                #     print(Fore.GREEN + "rollout")
                #     s = time.time()
                #     valid_states = []
                #     valid_nodes = []
                #     invalid_idx = []
                #     for i, state in enumerate(child_states):
                #         if not state.is_push_over():
                #             valid_states.append(state)
                #             valid_nodes.append(child_nodes[i])
                #         else:
                #             invalid_idx.append(i)
                #     if len(valid_states) > 0:
                #         rewards = self.root.batch_rollout(valid_nodes, valid_states, self.num_envs)
                #     e = time.time()
                #     print(len(valid_states), e - s)

                # # Backup TODO: assume the max depth of tree is 3
                # print(Fore.GREEN + "backup")
                # s = time.time()
                # if rewards is not None:
                #     for reward, node in zip(rewards, rollout_nodes):
                #         node.backpropagate(reward)
                #     for pi in invalid_idx:
                #         child_nodes[pi].backpropagate(child_nodes[pi].state.push_result)
                # else:
                #     for child_node in child_nodes:
                #         child_node.backpropagate(child_node.state.push_result)
                # e = time.time()
                # print(len(child_nodes), e - s)

                curr_time = time.time()
                duration = curr_time - start_time
                print(Fore.GREEN + f"Iteration: {itr}; Time: {duration:.2f} / {self.time_limit} s")
                itr += 1

        curr_time = time.time()
        duration = curr_time - start_time
        print(Fore.GREEN + f"Total Time: {duration:.2f} / {self.time_limit} s")

        return self.root.best_child(c_param=0, virtual_cost=0)

    # def _tree_policy(self):
    #     current_node = self.root
    #     while not current_node.is_terminal_node:
    #         if not current_node.is_fully_expanded:
    #             expanded, node = current_node.expand()
    #             if expanded:
    #                 return node
    #         else:
    #             if current_node.has_children:
    #                 current_node = current_node.best_child()
    #     return current_node

    def _parallel_tree_policy(self, active_nodes):
        candidate_nodes = []
        candidate_actions = []
        nodes_to_be_clean = []
        active_nodes_copy = active_nodes.copy()

        while len(candidate_actions) < self.num_envs:
            # If all active_nodes have been selected, we should quit
            is_fully_selected = True
            to_be_delete = []
            for node in active_nodes_copy:
                if node.state.uid not in node.state.mcts_helper.move_recorder and not node.state.is_push_over:
                    is_fully_selected = False
                    break
                elif not node.is_terminal_node and not node.is_fully_expanded:
                    is_fully_selected = False
                    break
                else:
                    to_be_delete.append(node)
            if is_fully_selected:
                break
            for node in to_be_delete:
                active_nodes_copy.remove(node)

            # Select a node
            current_node = self.root
            while not current_node.is_terminal_node:
                if not current_node.is_fully_expanded:
                    break
                else:
                    if current_node.has_children:
                        current_node = current_node.best_child()

            if current_node.is_fully_expanded or current_node.is_terminal_node:
                current_node.backpropagate_virtual(1)  # encourage to explore other nodes
            else:
                candidate_nodes.append(current_node)
                action = current_node.untried_actions.pop()
                candidate_actions.append(action)
                current_node.backpropagate_virtual(1)
            nodes_to_be_clean.append(current_node)

        return candidate_nodes, candidate_actions, nodes_to_be_clean

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

    # def _find_n_best_nodes(self, num=50):
    #     candidate_nodes = []
    #     while len(candidate_nodes) < num:
    #         node = self._tree_policy()
    #         if node not in candidate_nodes:
    #             candidate_nodes.append(node)
    #         # simple virtual loss
    #         node.backpropagate_virtual()

    #     # Expand
    #     self.root.level_expand(candidate_nodes)

    #     # Clean virtual visits
    #     for node in candidate_nodes:
    #         node.clean_virtual_visits()

    # def _get_all_not_fully_expanded_at_level(self, level):
    #     early_stop_check_nodes = []
    #     queue = self.root.children.copy()
    #     while len(queue) != 0:
    #         curr_node = queue.pop(0)
    #         if curr_node.state.level == level:
    #             if not curr_node.is_terminal_node and not curr_node.is_fully_expanded:
    #                 early_stop_check_nodes.append(curr_node)
    #         elif curr_node.state.level < level:
    #             queue.extend(curr_node.children.copy())

    #     return early_stop_check_nodes
