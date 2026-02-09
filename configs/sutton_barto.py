from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.ucb_agent import UCBAgent
from agents.gradient_agent import GradientAgent
from agents.constants import StepSizeMode, BaselineMode
from environments.stationary_gaussian_bandit import StationaryGaussianBandit
from environments.drifting_gaussian_bandit import DriftingGaussianBandit
from dataclasses import dataclass

# --- Configs ---

epsilon_greedy_agent_eps_01_sample_average_step_size_config = {'name': "Epsilon-greedy (ε = 0.1), Sample average step size mode", 
                                                                'class': EpsilonGreedyAgent, 
                                                                'params': {'epsilon': 0.1, 
                                                                           'optimistic_initialization' : False,
                                                                           'step_size_mode': StepSizeMode.SAMPLE_AVERAGE}
                                                                }

epsilon_greedy_agent_eps_01_constant_step_size_config = {'name': "Epsilon-greedy (ε = 0.1), step size is constant", 
                                                                'class': EpsilonGreedyAgent, 
                                                                'params': {'epsilon': 0.1, 
                                                                           'optimistic_initialization' : False,
                                                                           'step_size_mode': StepSizeMode.CONSTANT,
                                                                           'step_size': 0.1}
                                                                }

epsilon_greedy_agent_eps_001_sample_average_step_size_config = {'name': "Epsilon-greedy (ε = 0.01)", 
                                                                'class': EpsilonGreedyAgent, 
                                                                'params': {'epsilon': 0.01, 
                                                                           'optimistic_initialization' : False,
                                                                           'step_size_mode': StepSizeMode.SAMPLE_AVERAGE}
                                                                }

epsilon_greedy_agent_eps_0_sample_average_step_size_config = {'name': "Epsilon-greedy (ε = 0)", 
                                                                'class': EpsilonGreedyAgent, 
                                                                'params': {'epsilon': 0, 
                                                                           'optimistic_initialization' : False,
                                                                           'step_size_mode': StepSizeMode.SAMPLE_AVERAGE}
                                                              }

epsilon_greedy_agent_eps_00_constant_step_size_optimistic_init_config = {'name': "Epsilon-greedy (ε = 0), Constant step size mode and optimistic initialization", 
                                                                         'class': EpsilonGreedyAgent, 
                                                                         'params': {'epsilon': 0, 
                                                                                    'optimistic_initialization' : True,
                                                                                    'step_size_mode': StepSizeMode.CONSTANT,
                                                                                    'step_size': 0.1}
                                                                }

ucb_agent_c_2_sample_average = {'name': "UCB (c = 2)", 
                                'class': UCBAgent, 
                                 'params': {'exploration_degree': 2, 
                                            'optimistic_initialization' : False,
                                            'step_size_mode': StepSizeMode.SAMPLE_AVERAGE}}


gradient_agent_alpha_01_sample_average_baseline = {'name': "Gradient Agent (α = 0.1), sample average as baseline", 
                                                   'class': GradientAgent, 
                                                   'params': {'step_size' : 0.1,
                                                              'baseline_mode': BaselineMode.SAMPLE_AVERAGE}}

gradient_agent_alpha_04_sample_average_baseline = {'name': "Gradient Agent (α = 0.4), sample average as baseline", 
                                                   'class': GradientAgent, 
                                                   'params': {'step_size' : 0.4,
                                                              'baseline_mode': BaselineMode.SAMPLE_AVERAGE}}

gradient_agent_alpha_01_constant_baseline = {'name': "Gradient Agent (α = 0.1), without baseline", 
                                                   'class': GradientAgent, 
                                                   'params': {'step_size' : 0.1,
                                                              'baseline_mode': BaselineMode.CONSTANT,
                                                              'baseline_value': 0.00}}

gradient_agent_alpha_04_constant_baseline = {'name': "Gradient Agent (α = 0.4), without baseline", 
                                                   'class': GradientAgent, 
                                                   'params': {'step_size' : 0.4,
                                                              'baseline_mode': BaselineMode.CONSTANT,
                                                              'baseline_value': 0.00}}


stationary_standard_gaussian_bandit_config = {'class': StationaryGaussianBandit,
                                              'params':  {'n_arms' : 10,
                                                          'env_init_mean' : 0,
                                                          'env_init_variance' : 1,
                                                          'reward_variance' : 1},
                                              'is_stationary': True
                                             }

stationary_gaussian_bandit_mean_4_config = {'class': StationaryGaussianBandit,
                                              'params':  {'n_arms' : 10,
                                                          'env_init_mean' : 4,
                                                          'env_init_variance' : 1,
                                                          'reward_variance' : 1},
                                                'is_stationary': True
                                             }


non_stationary_standard_gaussian_bandit_config = {'class': DriftingGaussianBandit,
                                                  'params':  {'n_arms' : 10,
                                                              'drift_mean': 0,
                                                              'drift_var' : 0.01,
                                                              'reward_variance' : 1},
                                                  'is_stationary': False
                                             }

# --- Experiment Runtime Settings ---
@dataclass(frozen = True)
class SimulationConfig:
    """Holds the runtime parameters for the experiments."""
    N_RUNS: int
    T_STEPS_STATIONARY: int
    T_STEPS_NON_STATIONARY: int
    RANDOM_SEED: int = 2048

# --- Instantiation ---
global_sim_config = SimulationConfig(
    N_RUNS = 2000,
    T_STEPS_STATIONARY = 1000,
    T_STEPS_NON_STATIONARY = 10000
)


'''
Config for Exercise 2.5 S&B: Testing sample average step size versus constant in the non stationary environment
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
config_sample_averages_vs_constant_alpha_in_non_stationary_env = {"environment_prototype": non_stationary_standard_gaussian_bandit_config,
                                                                
                                                                "agent_prototypes": [epsilon_greedy_agent_eps_01_sample_average_step_size_config,
                                                                                     epsilon_greedy_agent_eps_01_constant_step_size_config]}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
Config for S&B Paragraph 2.3: testing different epsilon greedy algorithms with sample average step size and stationary environment
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
config_stationary_env_exploration_vs_exploitation_or_balance = {"environment_prototype": stationary_standard_gaussian_bandit_config,
                                                                
                                                                "agent_prototypes": [epsilon_greedy_agent_eps_01_sample_average_step_size_config,
                                                                                     epsilon_greedy_agent_eps_001_sample_average_step_size_config,
                                                                                     epsilon_greedy_agent_eps_0_sample_average_step_size_config]}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
Config for S&B Paragraph 2.6: Testing Optimistic initial values
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
config_optimistic_initial_values_vs_realistic = {"environment_prototype": stationary_standard_gaussian_bandit_config,
                                                 "agent_prototypes": [epsilon_greedy_agent_eps_01_constant_step_size_config,
                                                                      epsilon_greedy_agent_eps_00_constant_step_size_optimistic_init_config]}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
Config for S&B Section 2.7: Upper-Confidence-Bound Action Selection
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
config_ucb_vs_epsilon_greedy_stationary_env_sample_average_step_size = {"environment_prototype": stationary_standard_gaussian_bandit_config,
                                                                        "agent_prototypes": [epsilon_greedy_agent_eps_01_sample_average_step_size_config,
                                                                                             ucb_agent_c_2_sample_average]}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
Config for S&B Section 2.8: Gradient Bandit Algorithms
'''
config_gradient_agents_with_and_without_baselines = {"environment_prototype": stationary_gaussian_bandit_mean_4_config,
                                                     "agent_prototypes": [gradient_agent_alpha_01_sample_average_baseline,
                                                                          gradient_agent_alpha_04_sample_average_baseline,
                                                                          gradient_agent_alpha_01_constant_baseline,
                                                                          gradient_agent_alpha_04_constant_baseline
                                                                          ]}