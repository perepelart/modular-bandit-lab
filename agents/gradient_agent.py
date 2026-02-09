import numpy as np
from agents.base_agent import BaseAgent
from agents.constants import BaselineMode

from agents.agents_mixins.gradient_agent_plotting_mixin import GradientAgentPlottingMixin

class GradientAgent(BaseAgent, GradientAgentPlottingMixin):
    def __init__(self, 
                 step_size : float, 
                 baseline_mode: BaselineMode = BaselineMode.SAMPLE_AVERAGE,
                 baseline_value: float = 0.00):
        self.baseline_mode = baseline_mode

        self._init_config = {'step_size' : step_size,   
                             'baseline_mode': baseline_mode,
                             'baseline_value': baseline_value}

        
        super().__init__()  

        if self.baseline_mode is BaselineMode.CONSTANT:
            self.baseline_value = baseline_value
        elif self.baseline_mode is BaselineMode.SAMPLE_AVERAGE:
            self.baseline_value = 0.00
        else:
            raise NotImplementedError

        self.step_size = step_size
        self.action_preferences = None
        self.action_probabilities = None

    def initialize(self, environment):
        """
        Connects the agent to the environment, initializing Q-values and other state.
        """
        super().initialize(environment) # This sets self.number_of_arms  
        self.action_preferences = np.zeros((self.number_of_arms, ), dtype = np.float64) #H_t(a)
        # Initialize pi_t(a) with a uniform probability distribution
        self.action_probabilities = np.ones(self.number_of_arms) / self.number_of_arms



    def select_action(self):
        # Select an action probabilistically according to the learned policy
        return np.random.choice(self.number_of_arms, p = self.action_probabilities)
    
    def update_policy(self, A_t, R_t, t):
        self.__update_preferences(A_t, R_t)
        self.__update_action_probabilities()
        self.__update_baseline(R_t, t)

    def __update_preferences(self, A_t, R_t):
        self.action_preferences[A_t] += self.step_size * (R_t  - self.baseline_value) * (1 - self.action_probabilities[A_t])
         
        mask = np.ones(self.action_preferences.shape, dtype = bool)
        mask[A_t] = False
        self.action_preferences[mask] -= self.step_size * (R_t  - self.baseline_value) * self.action_probabilities[mask]

    def __update_action_probabilities(self):
        stable_preferences = self.action_preferences - np.max(self.action_preferences) # For numerical stability
        exp_preferences = np.exp(stable_preferences)
        self.action_probabilities = exp_preferences / np.sum(exp_preferences)

    def __update_baseline(self, R_t, t):
        if self.baseline_mode is BaselineMode.SAMPLE_AVERAGE:
            self.baseline_value += 1.0/t * (R_t - self.baseline_value)
        else:
            return None
        
    def clone(self):
        return GradientAgent(**self._init_config)




