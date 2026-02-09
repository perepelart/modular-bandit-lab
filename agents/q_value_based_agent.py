import numpy as np

from agents.base_agent import BaseAgent
from agents.constants import StepSizeMode

class QValueBasedAgent(BaseAgent):
    optimistic_constant = 5.0

    def __init__(self, 
                 optimistic_initialization : bool, 
                 step_size_mode: StepSizeMode,
                 step_size : float = None):
        super().__init__()
        
        self.optimistic_initialization = optimistic_initialization
        
        if self.optimistic_initialization:
            #need to understand how to choose this hyperparameter more intelligently, now just use the value S&B used
            self.optimistic_constant = 5.0
        else:
            self.optimistic_constant = None

        self.Q_values = None
        self.action_counts = None

        self.step_size_mode = step_size_mode
        self.step_size = step_size
        if self.step_size_mode is StepSizeMode.UNBIASED_TRICK:
            # Note: Unbiased Constant Step-Size Trick logic is defined 
            # but not yet active in this initialization block.
            raise NotImplementedError

    def initialize(self, environment):
        """
        Connects the agent to the environment, initializing Q-values and other state.
        """
        super().initialize(environment) # This sets self.number_of_arms
        self.Q_values = self.__init_q_values()
        self.action_counts = np.zeros((self.number_of_arms, ), dtype = np.int64) # For 1/N(A) step size



    def __init_q_values(self):
        if self.optimistic_initialization:
            Q_values = np.array([self.optimistic_constant] * self.number_of_arms, dtype = np.float64)
        else:
            Q_values = np.zeros((self.number_of_arms, ), dtype = np.float64)

        return Q_values
    
    def update_policy(self, action, reward, time_step = None):
        self.__update_q_values(action, reward)

    def __update_q_values(self, A_t, R_t):
        self.action_counts[A_t] += 1
                
        step_size = self.__step_size(A_t)
                
        self.Q_values[A_t] += step_size * \
                (R_t - self.Q_values[A_t])
        
    def __step_size(self, action_taken_by_agent):
        alpha = self.step_size

        match self.step_size_mode:
            case StepSizeMode.SAMPLE_AVERAGE: 
                step_size = 1.0 / self.action_counts[action_taken_by_agent]
            case StepSizeMode.CONSTANT:
                step_size = alpha
            case StepSizeMode.UNBIASED_TRICK:
                # Recursive calculation for trace o_n would go here
                raise NotImplementedError("Unbiased trick not yet fully implemented.")
            case _:
                raise NotImplementedError
        
        return step_size
