from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    An abstract base class for a multi-armed bandit agent.
    An instance of a Bandit represents a single run in an experiment.
    """
    def __init__(self):
        self.number_of_arms = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self, environment):
        """Initializes agent's state based on the environment's properties."""
        self.number_of_arms = environment.n_arms
        self.is_initialized = True

    @abstractmethod
    def select_action(self) -> int:
        pass

    @abstractmethod
    def update_policy(self, action: int, reward: float, time_step : int):
        pass

    @abstractmethod
    def clone(self):
        pass