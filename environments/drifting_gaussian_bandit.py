import numpy as np

from environments.base_bandit_environment import BaseBanditEnvironment
from environments.environments_mixins.drifting_gaussian_bandit_mixin import DriftingGaussianBanditMixin

# --- Non-Stationary Environments ---

class DriftingGaussianBandit(BaseBanditEnvironment, DriftingGaussianBanditMixin):
    """
    A non-stationary bandit where the true mean rewards q_*(a) for each
    arm take a random walk at each time step.
    """
    def __init__(self, n_arms: int, drift_mean: float = 0.0, drift_var: float = 0.01, reward_variance: float = 1.0):
        super().__init__(n_arms)
        self.drift_mean = drift_mean
        self.drift_variance = drift_var
        self.reward_variance = reward_variance
        self.reset()

    def reset(self):
        """Initializes the q_*(a) values to zero."""
        '''all the q^*(a) start out equal and then take independent random walks '''
        self._q_star = np.zeros(self.n_arms)
        self.optimal_action = np.argmax(self._q_star)

    def sample_reward(self, action: int) -> float:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.n_arms-1}.")
        # Sample reward
        reward = np.random.normal(loc = self._q_star[action], scale = np.sqrt(self.reward_variance))
        return reward
    
    #reward drifts
    def step(self) -> None:
        drift = np.random.normal(loc = self.drift_mean, scale = np.sqrt(self.drift_variance), size=self.n_arms)
        self._q_star += drift
        self.optimal_action = np.argmax(self._q_star)
    
    @property
    def is_stationary(self) -> bool:
        return False        