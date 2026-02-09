import numpy as np

from agents.q_value_based_agent import QValueBasedAgent
from agents.constants import StepSizeMode

from agents.agents_mixins.ucb_plotting_mixin import UCBPlottingMixin

class UCBAgent(QValueBasedAgent, UCBPlottingMixin):
    def __init__(self, exploration_degree, optimistic_initialization, step_size_mode : StepSizeMode, step_size = None):
        self._init_config = {'exploration_degree': exploration_degree, 
                             "optimistic_initialization": optimistic_initialization, 
                             'step_size_mode': step_size_mode,
                             'step_size' : step_size}

        super().__init__(optimistic_initialization, step_size_mode, step_size)
        self.exploration_degree = exploration_degree
        #time step counter
        self.t = 0

    def initialize(self, environment):
        """
        Connects the agent to the environment, initializing Q-values and other state.
        """
        super().initialize(environment)

    def select_action(self) -> int:
        untried_actions = np.where(self.action_counts == 0)[0]
        self.t += 1

        if untried_actions.size > 0:
             action_taken_by_agent = untried_actions[0]
        else:
            exploration_term = self.exploration_degree * np.sqrt(np.log(self.t) / self.action_counts)
            ucb_values = self.Q_values + exploration_term
            action_taken_by_agent = np.argmax(ucb_values)
        
        return action_taken_by_agent

    def update_policy(self, action: int, reward: float, time_step = None):
        return super().update_policy(action, reward)
    
    def clone(self):
        return UCBAgent(**self._init_config)
    
    