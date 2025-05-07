# SC3000-Artificial-Intelligence
# Reinforcement Learning for CartPole-v1 Control (SC3000 AI Project)

This project, part of the SC3000 Artificial Intelligence course at Nanyang Technological University, explores the application of various Reinforcement Learning (RL) algorithms to solve the classic CartPole-v1 control problem from the Gymnasium environment. The goal was to train agents capable of consistently achieving an average cumulative reward greater than 195 over 100 evaluation episodes.

## Table of Contents
* [Project Overview](#project-overview)
* [Algorithms Implemented](#algorithms-implemented)
* [Key Features & Contributions](#key-features--contributions)
* [Technologies Used](#technologies-used)
* [Setup and Installation](#setup-and-installation)
* [Running the Experiments](#running-the-experiments)
* [Results Summary](#results-summary)
* [Key Learnings](#key-learnings)
* [Author](#author)

## Project Overview

The CartPole-v1 problem involves balancing a pole on a cart that moves along a frictionless track. This project implements and compares three distinct RL agents: Q-Learning, Monte Carlo Control, and Deep Q-Networks (DQN), including an improved version (Double DQN) for the DQN agent and refined configurations for tabular methods.

## Algorithms Implemented

1.  **Q-Learning:**
    *   Tabular, model-free, off-policy Temporal Difference (TD) method.
    *   Required state discretization of the continuous observation space.
    *   Compared original (10x10x10x10 bins) vs. improved (20x20x20x20 bins) configurations.
2.  **Monte Carlo (MC) Control:**
    *   Tabular, model-free method learning from complete episodes (First-Visit MC).
    *   Also utilized state discretization.
    *   Compared original vs. improved configurations with increased bins and significantly more training episodes.
3.  **Deep Q-Network (DQN):**
    *   Function approximation method using a neural network (PyTorch) to estimate Q-values from continuous states.
    *   Implemented with Experience Replay and a Target Network.
    *   Compared standard DQN vs. an improved version implementing Double DQN (DDQN) with hyperparameter tuning.

## Key Features & Contributions

*   Implementation of `QLearningAgent`, `MonteCarloAgent`, and `DQNAgent` classes.
*   State discretization logic for tabular methods.
*   `QNetwork` and `ReplayMemory` components for DQN.
*   Extensible experimental framework for systematic training, evaluation, and comparison of multiple agent configurations.
*   Detailed analysis of agent performance, training times, and the impact of specific improvements (e.g., bin size, DDQN).
*   Visualization of training progress and evaluation results.

## Technologies Used

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Gymnasium (for the CartPole-v1 environment)
    *   PyTorch (for Deep Q-Networks)
    *   NumPy (for numerical operations)
    *   Pandas (for results display and analysis)
    *   Matplotlib (for plotting)
*   **Development Environment:** Jupyter Notebook
## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone [Your GitHub Repo URL]
    cd [repository-name]
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install gymnasium[classic_control] pygame torch numpy matplotlib pandas ipython
    ```
 

## Running the Experiments

The main experimental script/notebook is `[SC3000_TinJingLunJavier_SDAE_StillPondTeam_Submission.ipynb]`.
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook [SC3000_TinJingLunJavier_SDAE_StillPondTeam_Submission.ipynb]
    ```
2.  Run the cells sequentially to:
    *   Set up the environment and dependencies.
    *   Define agent classes and helper functions.
    *   Execute Task 1 (Agent Development Demonstration).
    *   Execute Task 2 (Agent Effectiveness Demonstration - Original vs. Improved for QL, MC, DQN). This will train and evaluate all configurations.
    *   Execute Task 3 (Render an episode with the best agent).
    *   View Task 4 (Analysis and Conclusions).



## Results Summary



*   **Best Performing Agent:** Improved Monte Carlo (MC_Improved) with an average reward of **445.57**.
*   **Standard DQN (DQN_Original):** Successfully solved the task with an average reward of **217.32**.
*   Tabular methods demonstrated significant improvement with finer discretization and more training.
*   The "Improved DQN" (DDQN) configuration underperformed, highlighting DQN's hyperparameter sensitivity.



## Key Learnings

*   Practical implementation nuances of Q-Learning, Monte Carlo, and DQN algorithms.
*   Importance of state representation (discretization) for tabular RL methods.
*   The trade-offs between sample efficiency and final performance (e.g., DQN vs. MC).
*   Challenges and sensitivity of hyperparameter tuning in Deep Reinforcement Learning.
*   Techniques for stabilizing DQN training (Experience Replay, Target Networks, Double DQN).

## Author

*   Javier Tin Jing Lun ([Javier2417])
