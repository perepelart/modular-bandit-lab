import numpy as np

from agents.q_value_based_agent import QValueBasedAgent
from agents.agents_mixins.epsilon_greedy_plotting_mixin import EpsilonGreedyPlottingMixin
from agents.constants import StepSizeMode

class EpsilonGreedyAgent(QValueBasedAgent, EpsilonGreedyPlottingMixin):
    def __init__(self, epsilon, optimistic_initialization, step_size_mode : StepSizeMode, step_size = None):
        self._init_config = {'epsilon': epsilon, 
                             "optimistic_initialization": optimistic_initialization, 
                             'step_size_mode': step_size_mode,
                             'step_size' : step_size}
        super().__init__(optimistic_initialization, step_size_mode, step_size)
        self.epsilon = epsilon

    def initialize(self, environment):
        """
        Connects the agent to the environment, initializing Q-values and other state.
        """
        super().initialize(environment)

# --- Implementation of the core logic ---

    def select_action(self):
        decision = np.random.choice(a = ['explore', 'exploit'], 
                                    p = [self.epsilon, 1 - self.epsilon])
        if decision == 'explore':
            action_taken_by_agent = np.random.randint(low = 0, high = self.number_of_arms)
        else:
            max_q = np.max(self.Q_values)
            best_actions = np.where(self.Q_values == max_q)[0]
            action_taken_by_agent = np.random.choice(best_actions)
        return action_taken_by_agent
    
    def update_policy(self, action, reward, time_step = None):
        return super().update_policy(action, reward, time_step)
    
    def clone(self):
        return EpsilonGreedyAgent(**self._init_config)
    
