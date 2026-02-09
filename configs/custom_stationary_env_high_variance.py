from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.ucb_agent import UCBAgent
from agents.constants import StepSizeMode
from environments.stationary_gaussian_bandit import StationaryGaussianBandit
from dataclasses import dataclass

"""
Custom Experiment Configuration: High Variance Stress Test

Motivation:
Standard bandit benchmarks often assume unit variance (sigma^2 = 1). 
In real-world scenarios (financial markets, sensor data), noise can significantly 
overpower the signal (mean reward).

This configuration defines a 'High Variance' environment (sigma^2 = 10) to test 
the robustness of UCB vs. Epsilon-Greedy agents when the signal-to-noise ratio is low.
"""

# --- Environment Config ---

high_variance_stationary_bandit_config = {
    'class': StationaryGaussianBandit,
    'params': {
        'n_arms': 10,
        'env_init_mean': 0,
        'env_init_variance': 1,
        'reward_variance': 10.0  # 10x higher variance than standard S&B
    },
    'is_stationary': True
}

# --- Agent Configs ---

# Re-using standard agents to see how they degrade under stress
epsilon_greedy_standard = {
    'name': "Epsilon-greedy (ε = 0.1)", 
    'class': EpsilonGreedyAgent, 
    'params': {
        'epsilon': 0.1, 
        'optimistic_initialization': False,
        'step_size_mode': StepSizeMode.SAMPLE_AVERAGE
    }
}

ucb_agent_standard = {
    'name': "UCB (c = 2)", 
    'class': UCBAgent, 
    'params': {
        'exploration_degree': 2, 
        'optimistic_initialization': False,
        'step_size_mode': StepSizeMode.SAMPLE_AVERAGE
    }
}

# --- Experiment Definition ---

config_high_variance_comparison = {
    "environment_prototype": high_variance_stationary_bandit_config,
    "agent_prototypes": [epsilon_greedy_standard, ucb_agent_standard]
}

# --- Runtime Settings ---

@dataclass(frozen=True)
class CustomSimulationConfig:
    N_RUNS: int = 2000
    T_STEPS: int = 1000

custom_sim_config = CustomSimulationConfig()