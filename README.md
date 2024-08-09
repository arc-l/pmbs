# Parallel Monte Carlo Tree Search with Batched Rigid-body Simulations for Speeding up Long-Horizon Episodic Robot Planning

**Abstract.** We propose a novel Parallel Monte Carlo tree search with Batched Simulations (PMBS) algorithm for accelerating long-horizon, episodic robotic planning tasks. Monte Carlo tree search (MCTS) is an effective heuristic search algorithm for solving episodic decision-making problems whose underlying search spaces are expansive. Leveraging a GPU-based large-scale simulator, PMBS introduces massive parallelism into MCTS for solving planning tasks through the batched execution of a large number of concurrent simulations, which allows for more efficient and accurate evaluations of the expected cost-to-go over large action spaces. When applied to the challenging manipulation tasks of object retrieval from clutter, PMBS achieves a speedup of over 30 $\times$ with an improved solution quality, in comparison to a serial MCTS implementation. We show that PMBS can be directly applied to real robot hardware with negligible sim-to-real differences.

[YouTube (presentation)](https://youtu.be/-Br2IBjArgY)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/abs/2207.06649)&nbsp;&nbsp;•&nbsp;&nbsp;International Conference on Intelligent Robots and Systems (IROS) 2022

*[Baichuan Huang](https://baichuan05.github.io/), [Abdeslam Boularias](http://rl.cs.rutgers.edu/abdeslam.html), [Jingjin Yu](http://jingjinyu.com/)*

## 1 minutue intro video:

https://user-images.githubusercontent.com/20850928/157065260-b8fc6c1f-7241-4fe2-a372-ae480fe17ae8.mp4


## Installation (developed on Ubuntu 18.04, also tested on Ubuntu 20.04)
1. Download isaac gym preview 4 (developed on preview 3, also tested on preview 4) from Nvidia https://developer.nvidia.com/isaac-gym.
2. Follow the installation guide from issac gym (in directory of issac gym)
   1. `./create_conda_env_rlgpu.sh`
   2. `conda activate rlgpu`
   3. `export LD_LIBRARY_PATH=/home/xyz/anaconda3/envs/rlgpu/lib` If you are running Ubuntu 20.04
3. Install extra packages
   1. `pip install opencv-python==4.5.4.60 graphviz==0.19.1 termcolor colorama pandas pybullet==3.2.4 pynvml`
4. `mkdir logs_mcts`
5. Models can be accessed from https://drive.google.com/drive/folders/10VdC2ur7beE1yhmCBal3Ftw8Uhzlq_BH?usp=sharing
6. In case pytorch errors, try `pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111`

## Quick Start (benchmarking as presented in the paper)
NOTE: We tested on RTX 2080 Ti and RTX A4000. The memory of GPU should be large, if not, you should use a smaller number of environments (look into shell script).

### MCTS-30 (baseline)
* Run `bash mcts_run.sh`
* Put all logs in a single folder (inside of `logs_mcts`).
* Run `python evaluate.py --log 'PATH_TO_FOLDER_OF_MCTS_RECORDS'` to get benchmark result.
* We have two environments, the first one is for mimicing the real-world environemnt and the second one is for planning.

### PMBS-30 (proposed, 500 environments)
* Run `bash mcts_parallel_run.sh`
* Put all logs in a single folder (inside of `logs_mcts`).
* Run `python evaluate.py --log 'PATH_TO_FOLDER_OF_MCTS_RECORDS'` to get benchmark result.
* We have 500 environments, the first one is for mimicing the real-world environemnt and all others are for planning.

With GUI on, you should expect to see something like this (video has been shortened as the rendering slow down the simulation):

https://user-images.githubusercontent.com/20850928/197361606-9d903f67-0e0b-47a5-961c-bd3d62ca9a87.mp4

## Train Grasp Classifier 
1. Grasp Classifier data collection please refer to collect_grasp_data.py. Generate files first, collect labeled data after.
2. Training `python train_grasp.py --dataset_root 'logs_grasp'`

## Others
* Run `python action_utils.py` to pre-record possible actions for each type of objects.
* Grasp Network (DQN) is taken from https://github.com/arc-l/more.

## Real robot experiments (videos):
All experiments are executed under a time budget of 30 seconds (we gave 30 seconds for planning). We end the experiment if the robot cannot solve the task within 16 actions.
1. parallel MCTS: https://drive.google.com/drive/folders/1D2WH-WXGmlJ-0r_p061nJd_sqZesR_2k?usp=sharing
2. serial MCTS: https://drive.google.com/drive/folders/1D2blIxYsoRmiuw0keBcCSqaxri-i38wv?usp=sharing

The parallel version is stable (low variance), different runs have similar result. 
The serial version is unstable (high variance), it could solve the problem very well, or not at all. This is due to the randomness of the tree policy.

## Supplementary material
<!-- [PMBS_Supplementary.pdf](https://github.com/arc-l/pmbs/files/8199209/PMBS_Supplementary.pdf) -->
[PMBS_Supplementary.pdf](PMBS_Supplementary.pdf)

We are trying to solve the task using the physics simulator. The sim-to-real gap is there, but it is good enough for this type of task (even with pose estimation error).

The simulator could provide accurate physics simulations:

<img src="https://user-images.githubusercontent.com/20850928/157085758-4f106057-ecbb-4ae8-a568-c524454343b3.png" width="600">

The simulator could provide not so accurate but reasonable physics simulations:

<img src="https://user-images.githubusercontent.com/20850928/157085771-37fbaeb0-37cc-4b95-8f62-d1dbe69e07bd.png" width="900">


## Citing this paper
If this work helps your research, please cite the [PMBS](https://arxiv.org/abs/2207.06649):

```
@inproceedings{huang2022parallel,
  title        = {Parallel Monte Carlo Tree Search with Batched Rigid-body Simulations for Speeding up Long-Horizon Episodic Robot Planning},
  author       = {Huang, Baichuan and Boularias, Abdeslam and Yu, Jingjin},
  booktitle    = {The IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year         = {2022},
  organization = {IEEE}
}
```
