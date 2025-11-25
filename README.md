\# Deep Q-Network for Antenna Architecture Optimization



This project implements a \*\*Deep Q-Network (DQN)\*\* to automatically optimize antenna array architectures based on a pre-generated dataset.  

The agent learns to select the best antenna configuration (number of elements per ring) in order to maximize RF performance metrics such as main-lobe gain, side-lobe level (SSL) and HPBW.



---



\## 📂 Project Structure





.

├── Dataset

│ ├── Minput.npy # RF performance vectors \[MainLobe, SSL, HPBW, Theta0]

│ ├── Moutput.npy # Architecture vectors (elements per ring)

│ └── dataset.py # Dataset generation script (optional)

│

├── Qmodel

│ ├── env\_dataset.py # Reinforcement Learning environment based on the dataset

│ ├── dqn\_agent.py # DQN agent, replay buffer and neural network

│ ├── train.py # Training loop

│ └── plot\_rewards.py # Script to plot reward evolution

│

└── README.md

---



\## 📊 Dataset Description



The dataset contains two matrices:



\### \*\*Minput (4 × N)\*\*  

RF performance obtained for each antenna configuration:



1\. \*\*Main Lobe Gain\*\*  

2\. \*\*Side Lobe Level (SSL)\*\*  

3\. \*\*Half-Power Beamwidth (HPBW)\*\*  

4\. \*\*Steering angle θ₀ (degrees)\*\*  



Each column corresponds to a different architecture.



---



\### \*\*Moutput (5 × N)\*\*  

Antenna architecture description:



\- Number of elements in \*\*Ring 1\*\*  

\- Number of elements in \*\*Ring 2\*\*  

\- Number of elements in \*\*Ring 3\*\*  

\- Number of elements in \*\*Ring 4\*\*  

\- Number of elements in \*\*Ring 5\*\*  



Each architecture is treated as one \*\*discrete action\*\* in the RL environment.



---



\## 🧠 Reinforcement Learning Environment



The environment is located in \*\*Qmodel/env\_dataset.py\*\*.



\### \*\*State (4-dimensional)\*\*  

A normalized vector:  

\\\[

s = \[MainLobe, SSL, HPBW, \\theta\_0]

\\]



\### \*\*Action\*\*  

An integer in \*\*\[0, N-1]\*\*, each representing one antenna architecture (column of Moutput).



\### \*\*Reward function\*\*

\\\[

r = MainLobe - |SSL| - HPBW

\\]

This encourages high main-lobe gain, low side-lobe level, and narrow beamwidth.



\### \*\*Episode structure\*\*

\- Multi-step pseudo-dynamics  

\- Default length = 10 steps  

\- Each action jumps to another sample from the dataset  

\- This creates a stable RL environment suitable for DQN training



---



\## 🤖 Deep Q-Network (DQN)



The DQN agent (in \*\*dqn\_agent.py\*\*) implements:



\- Feedforward neural network with two hidden layers  

\- Replay buffer  

\- Target network  

\- Epsilon-greedy exploration  

\- MSE loss for the Bellman target  

\- Adam optimizer  



It follows the standard DQN formulation.



---



\## 🚀 Training



Run the training script:



```bash

cd Qmodel

python train.py



