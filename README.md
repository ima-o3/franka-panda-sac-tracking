\# Franka Panda SAC End-Effector Tracking in MuJoCo



This project implements a reinforcement learning controller for 3D end-effector trajectory tracking using a Franka Panda robotic arm in MuJoCo. The main task is to track a circular Cartesian trajectory using a Soft Actor-Critic (SAC) agent, with comparison against a classical Jacobian inverse kinematics baseline.



The project also introduces uncertainty through observation noise, action noise, and unreachable target trajectories to evaluate the robustness of the learned controller.



\## Project Overview



The aim of this project is to control a 7-DOF Franka Panda arm so that its end-effector tracks a moving circular target trajectory in Cartesian space.



The project includes:



\* A custom MuJoCo + Gymnasium environment for circular trajectory tracking

\* A classical Jacobian inverse kinematics baseline

\* A Soft Actor-Critic reinforcement learning controller

\* Periodic model evaluation and best-checkpoint selection

\* Robustness testing with observation noise, action noise, and unreachable target positions

\* Tracking error plots and simulation videos



\## Demo



Add demo GIFs or videos here.



Example:



```markdown

!\[SAC Circular Tracking](results/videos/sac\_tracking.gif)

```



Or link to videos:



\* Clean SAC tracking demo: `results/videos/sac\_clean\_tracking.mp4`

\* Disturbed SAC tracking demo: `results/videos/sac\_noisy\_tracking.mp4`

\* Classical IK baseline demo: `results/videos/classical\_ik\_tracking.mp4`



\## Repository Structure



```text

franka-panda-sac-tracking/

│

├── circle\_ik\_panda.py

├── panda\_circle\_sac\_env.py

├── train\_sac\_circle.py

├── play\_sac\_circle.py

│

├── Noisy\_SAC/

│   ├── panda\_circle\_sac\_env\_uncertain.py

│   ├── train\_sac\_circle\_uncertain.py

│   └── play\_sac\_circle\_uncertain.py

│

├── Original\_SAC/

│   └── models\_clean/

│       └── best\_model/

│           └── best\_model.zip

│

├── results/

│   ├── plots/

│   └── videos/

│

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation



This project was developed on Windows using Python, MuJoCo, Gymnasium, and Stable-Baselines3.



\### 1. Clone this repository



```bash

git clone https://github.com/ima-o3/franka-panda-sac-tracking.git

cd franka-panda-sac-tracking

```



\### 2. Create and activate a virtual environment



On Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

pip install -r requirements.txt

```



\### 4. Download MuJoCo Menagerie



This project uses the Franka Panda model from MuJoCo Menagerie.



Clone MuJoCo Menagerie so that the folder structure looks like this:



```text

mujoco\_projects/

│

├── mujoco\_menagerie/

│   └── franka\_emika\_panda/

│       └── scene.xml

│

└── panda\_tracking\_rl/

&#x20;   └── project files

```



Clone Menagerie with:



```bash

git clone https://github.com/google-deepmind/mujoco\_menagerie.git

```



The environment expects the Franka Panda scene file at:



```text

mujoco\_menagerie/franka\_emika\_panda/scene.xml

```



\## How to Run



\### Classical Jacobian IK Baseline



```powershell

python circle\_ik\_panda.py

```



This runs a classical kinematic controller that tracks a circular trajectory using the end-effector Jacobian and damped least-squares inverse kinematics.



\### Train Clean SAC Agent



```powershell

python train\_sac\_circle.py

```



This trains a SAC policy on the clean circular tracking task.



\### Play Clean SAC Agent



```powershell

python play\_sac\_circle.py

```



This loads the best trained clean SAC policy and renders it in the MuJoCo viewer.



\### Train SAC Agent with Disturbances



```powershell

python Noisy\_SAC/train\_sac\_circle\_uncertain.py

```



This trains a SAC policy with observation noise, action noise, and probabilistic unreachable target episodes.



\### Play SAC Agent with Disturbances



```powershell

python Noisy\_SAC/play\_sac\_circle\_uncertain.py

```



This evaluates the disturbed SAC policy in the MuJoCo viewer.



\## Task Formulation



The task is formulated as a continuous-control Markov Decision Process.



\### State / Observation



The policy observes:



\* 7 Panda joint positions

\* 7 Panda joint velocities

\* 3D end-effector position

\* 3D target position

\* 3D Cartesian tracking error

\* Circular trajectory phase as sine and cosine features

\* Current commanded joint targets



The observation vector has dimension 32.



\### Action



The action is a 7D continuous vector in the range:



```text

\[-1, 1]

```



Each action represents an incremental update to the commanded joint position targets of the Panda arm.



The action is scaled before being applied:



```text

q\_cmd = q\_cmd + action \* action\_scale

```



\### Transition Dynamics



The transition dynamics are provided by MuJoCo. At every environment step, the selected action updates the joint target command, MuJoCo advances the simulation, and the next observation is returned.



\### Reward



The reward is designed to encourage accurate Cartesian tracking while discouraging aggressive or unstable motion.



The reward includes:



\* Cartesian tracking error penalty

\* Action effort penalty

\* Action change penalty

\* Joint velocity penalty

\* Bonuses for accurate tracking thresholds



The reward structure is:



```python

reward\_error = min(tracking\_error, 0.50)



reward = (

&#x20;   -10.0 \* reward\_error

&#x20;   -0.01 \* action\_effort

&#x20;   -0.02 \* action\_change

&#x20;   -0.001 \* joint\_speed

)



if tracking\_error < 0.08:

&#x20;   reward += 0.2



if tracking\_error < 0.05:

&#x20;   reward += 0.5



if tracking\_error < 0.03:

&#x20;   reward += 1.0

```



The tracking error is clipped in the reward for disturbed environments so that unreachable target episodes do not dominate training too aggressively.



\## Reinforcement Learning Method



The RL controller uses Soft Actor-Critic (SAC), an off-policy actor-critic algorithm for continuous action spaces.



SAC was selected because:



\* The Panda arm requires continuous joint-space control

\* SAC handles continuous actions naturally

\* It encourages exploration through entropy maximisation

\* It is sample-efficient compared with many on-policy methods

\* It is commonly used for robotic control problems



The model is trained using Stable-Baselines3.



During training, periodic evaluation is used to save the best model checkpoint. The final policy is not assumed to be the best policy, because RL performance can fluctuate during training.



\## Classical IK Baseline



A classical Jacobian inverse kinematics controller is implemented as a baseline.



The controller computes the desired end-effector velocity from the target circular trajectory and tracking error, then converts this Cartesian velocity into joint velocities using damped least-squares inverse kinematics:



```text

dq = Jᵀ (J Jᵀ + λ²I)⁻¹ v\_des

```



This provides a non-learning baseline for comparing the SAC policy against classical robotics control.



\## Disturbance and Robustness Testing



To evaluate robustness, the disturbed environment introduces three uncertainty sources.



\### Observation Noise



Gaussian noise is added to the observation received by the policy. This simulates imperfect sensing.



The true MuJoCo state is not changed; only the policy input is perturbed.



\### Action Noise



Gaussian noise is added to the action before execution. This simulates imperfect actuation or low-level control error.



The policy outputs a raw action, but the environment applies a noisy executed action.



\### Unreachable Target Episodes



Some episodes place the circular target outside the normal reachable workspace of the Panda arm.



This tests whether the controller can remain stable when perfect tracking is impossible.



The environment can be configured with:



```python

observation\_noise\_std=0.002

action\_noise\_std=0.05

unreachable\_prob=0.20

unreachable\_jitter\_std=0.03

```



\## Experiments



The following experiments were performed:



| Experiment    | Description                                                               |

| ------------- | ------------------------------------------------------------------------- |

| Random Policy | Sanity check with random actions                                          |

| Classical IK  | Jacobian inverse kinematics baseline                                      |

| Clean SAC     | SAC trained on reachable circular tracking                                |

| Disturbed SAC | SAC trained with observation noise, action noise, and unreachable targets |

| Stress Test   | Evaluation under combined uncertainty                                     |



\## Results



Add your final results table here.



Example format:



| Controller    | Mean Tracking Error \[m] | Final Tracking Error \[m] | Max Tracking Error \[m] | Notes                                         |

| ------------- | ----------------------: | -----------------------: | ---------------------: | --------------------------------------------- |

| Random Policy |                     TBD |                      TBD |                    TBD | Poor tracking sanity check                    |

| Classical IK  |                     TBD |                      TBD |                    TBD | Smooth kinematic baseline                     |

| Clean SAC     |                     TBD |                      TBD |                    TBD | Best checkpoint selected by evaluation reward |

| Disturbed SAC |                     TBD |                      TBD |                    TBD | Trained with noise and unreachable episodes   |



\## Example Plots



Add plots here:



```markdown

!\[Tracking Error](results/plots/tracking\_error.png)

!\[XY Trajectory](results/plots/xy\_trajectory.png)

```



Suggested plots:



\* Target vs actual end-effector XY path

\* Tracking error over time

\* Clean SAC vs disturbed SAC comparison

\* Classical IK vs SAC tracking comparison



\## Key Design Decisions



\### Why MuJoCo?



MuJoCo provides fast rigid-body simulation, stable contact/dynamics modelling, and standard support for robot learning environments.



\### Why Franka Panda?



The Franka Panda is a widely used 7-DOF robotic arm in manipulation and control research. Its redundancy makes it suitable for trajectory tracking and continuous-control experiments.



\### Why SAC?



SAC is well-suited to continuous action spaces and robotic control. It learns a stochastic policy during training but can be evaluated deterministically.



\### Why use best-checkpoint selection?



RL performance is not guaranteed to improve monotonically. In this project, the best-performing policy during training was selected using periodic evaluation rather than assuming the final checkpoint was optimal.



\### Why include a classical IK baseline?



The IK controller provides a robotics-based reference point. Comparing SAC against IK helps distinguish learning performance from basic task feasibility.



\### Why add uncertainty?



Real robotic systems involve imperfect sensing, imperfect actuation, and infeasible commands. Adding uncertainty makes the task more realistic and tests whether the learned policy remains stable under disturbances.



\## Limitations



Current limitations include:



\* The policy controls joint position targets rather than torques

\* The task currently focuses on position tracking, not full pose/orientation tracking

\* The unreachable target handling is simplified

\* The disturbance model is basic Gaussian noise rather than a full sensor/actuator model

\* Training is performed entirely in simulation

\* The reward function requires manual tuning



\## Future Work



Potential extensions include:



\* Add end-effector orientation tracking

\* Add control delay

\* Add domain randomisation

\* Compare SAC with PPO or TD3

\* Train with curriculum learning

\* Evaluate across multiple random seeds

\* Add obstacle avoidance

\* Use torque-level control

\* Improve reward shaping for unreachable targets

\* Export cleaner evaluation videos and GIFs



\## Requirements



Main Python packages:



```text

mujoco

gymnasium

stable-baselines3

numpy

matplotlib

tensorboard

pandas

```



Install with:



```bash

pip install -r requirements.txt

```



\## Author



Idris Muzaffar Ariff

MSc Advanced Mechanical Engineering

Imperial College London



