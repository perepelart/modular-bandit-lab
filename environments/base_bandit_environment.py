from abc import ABC, abstractmethod

class BaseBanditEnvironment(ABC):
    """
    An abstract base class for all N-Armed Bandit environments.
    It defines the interface that agents will interact with.
    """
    def __init__(self, n_arms: int):
        if n_arms <= 0:
            raise ValueError("Number of arms must be a positive integer.")
        self._n_arms = n_arms

    @property
    def n_arms(self) -> int:
        """Returns the number of arms in the environment."""
        return self._n_arms

    @abstractmethod
    def sample_reward(self, action: int) -> float:
        """
        Takes an action and returns the resulting reward.
        This method contains the core logic of the environment.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment to its initial state.
        Useful for starting a new run in an experiment.
        """
        pass

    @property
    @abstractmethod
    def is_stationary(self) -> bool:
        pass