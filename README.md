## 1 minutue intro video:

https://user-images.githubusercontent.com/20850928/157065260-b8fc6c1f-7241-4fe2-a372-ae480fe17ae8.mp4

Higher quality video can be found at https://drive.google.com/file/d/1A0D7_eRgr43-qnC6tVDFfvjeFU7L9T-v/view?usp=sharing

## Real robot experiments (videos):
All experiments are executed under a time budget of 30 seconds (we gave 30 seconds for planning). We end the experiment if the robot cannot solve the task within 16 actions.
1. parallel MCTS: https://drive.google.com/drive/folders/1D2WH-WXGmlJ-0r_p061nJd_sqZesR_2k?usp=sharing
2. serial MCTS: https://drive.google.com/drive/folders/1D2blIxYsoRmiuw0keBcCSqaxri-i38wv?usp=sharing

The parallel version is stable (low variance), different runs have similar result. 
The serial version is unstable (high variance), it could solve the problem very well, or not at all. This is due to the randomness of the tree policy.

## Supplementary material
[PMBS_Supplementary.pdf](https://github.com/arc-l/pmbs/files/8199209/PMBS_Supplementary.pdf)

We are trying to solve the task using the physics simulator. The sim-to-real gap is there, but it is good enough for this type of task (even with pose estimation error).

The simulator could provide accurate physics simulations:

<img src="https://user-images.githubusercontent.com/20850928/157085758-4f106057-ecbb-4ae8-a568-c524454343b3.png" width="600">

The simulator could provide not so accurate but reasonable physics simulations:

<img src="https://user-images.githubusercontent.com/20850928/157085771-37fbaeb0-37cc-4b95-8f62-d1dbe69e07bd.png" width="900">
