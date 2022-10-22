import os
import sys
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
# parser.add_argument('--file_reward', action='store', type=str)
# parser.add_argument('--file_action', action='store', type=str)
parser.add_argument("--log", action="store", type=str)
# parser.add_argument("--num", default=1, type=int, action="store")
args = parser.parse_args()

sub_roots = glob.glob(f"{args.log}/*")
sub_roots = sorted(sub_roots, key=lambda sub: sub.split("-")[-1])
total = []
completion = []
grasp_success = []
num_action = []
time = []
for sub in sub_roots:
    reward_file = os.path.join(sub, "transitions", "label-value.log.txt")
    action_file = os.path.join(sub, "transitions", "executed-action.log.txt")
    time_file = os.path.join(sub, "transitions", "planning-time.log.txt")

    reward_log = np.loadtxt(reward_file, delimiter=" ")
    action_log = np.loadtxt(action_file, delimiter=" ")
    time_log = np.loadtxt(time_file, delimiter=" ")

    print(sub)
    print(f"total action is: {len(reward_log)}")
    print(f"get the target object: {np.sum(reward_log == 1)}")
    print(f"average number: {len(reward_log) / len(time_log)}")
    total.append(len(reward_log) / len(time_log))
    completion.append(np.sum(reward_log == 1))
    num_action.append(len(reward_log) / len(time_log))

    assert len(reward_log) == len(action_log)
    action_log = action_log[:, 0]
    print(f"grasp success: {np.sum(reward_log[action_log == 1]) / np.sum(action_log == 1)}")
    print(reward_log[action_log == 1])
    grasp_success.append(np.sum(reward_log[action_log == 1]) / np.sum(action_log == 1))

    average_time = np.sum(time_log) / len(time_log)
    print(f"time: {time_log}, average: {average_time}")
    time.append(average_time)
print(sum(total) / len(total))
print(f"completion: {completion}")
print(f"grasp_success: {grasp_success}")
print(f"num_action: {num_action}")
print(f"time: {time}")
print("completion", sum(completion) / len(completion))
print("grasp success:", sum(grasp_success) / len(completion))
print("ave num of action:", sum(num_action) / len(num_action))
print("ave time:", sum(time) / len(time))

# new = os.path.isfile(os.path.join(args.log, "transitions", "label-value.log.txt"))
# print(args.log)

# if new:
#     reward_file = os.path.join(args.log, "transitions", "label-value.log.txt")
#     action_file = os.path.join(args.log, "transitions", "executed-action.log.txt")

#     reward_log = np.loadtxt(reward_file, delimiter=' ')
#     print(f"total action is: {len(reward_log)}")
#     print(f"get the target object: {np.sum(reward_log == 1)}")
#     print(f"average number: {len(reward_log) / np.sum(reward_log == 1)}")

#     action_log = np.loadtxt(action_file, delimiter=' ')
#     assert len(reward_log) == len(action_log)
#     action_log = action_log[:, 0]
#     print(f"grasp success: {np.sum(reward_log[action_log == 1]) / np.sum(action_log == 1)}")
#     print(reward_log[action_log == 1])
# else:
# reward_file = os.path.join(args.log, "transitions", "reward-value.log.txt")
# action_file = os.path.join(args.log, "transitions", "executed-action.log.txt")

# reward_log = np.loadtxt(reward_file, delimiter=' ')
# print(f"total action is: {len(reward_log)}")
# print(f"get the target object: {np.sum(reward_log == 1)}")
# print(f"average number: {len(reward_log) / np.sum(reward_log == 1)}")

# action_log = np.loadtxt(action_file, delimiter=' ')
# assert len(reward_log) == len(action_log)
# action_log = action_log[:, 0]
# print(f"grasp success: {np.sum(reward_log[action_log == 0]) / np.sum(action_log == 0)}")
# print(reward_log[action_log == 0])

