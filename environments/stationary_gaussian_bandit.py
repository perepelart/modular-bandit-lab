import numpy as np
from environments.base_bandit_environment import BaseBanditEnvironment
from environments.environments_mixins.stationary_gaussian_bandit_mixin import StationaryGaussianBanditMixin

# --- Stationary Stochastic Environments ---

class StationaryGaussianBandit(BaseBanditEnvironment, StationaryGaussianBanditMixin):
    """
    A classic stationary N-armed bandit where rewards for each arm 'a' are
    sampled from a normal distribution with mean q_*(a) and a fixed variance.
    """
    def __init__(self, n_arms: int, env_init_mean = 0, env_init_variance = 1, reward_variance: float = 1.0):
        super().__init__(n_arms)
        self.env_init_mean = env_init_mean
        self.env_init_variance = env_init_variance
        self.reward_variance = reward_variance
        self.reset()

    def reset(self):
        """Initializes the true mean reward q_*(a) for each arm."""
        # True action values are sampled from a standard normal distribution
        self._q_star = np.random.normal(loc = self.env_init_mean, scale = np.sqrt(self.env_init_variance), size = self.n_arms)
        self.optimal_action = np.argmax(self._q_star)

    def sample_reward(self, action: int) -> float:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.n_arms - 1}.")
        
        # Sample reward from N(q_*(action), reward_variance)
        reward = np.random.normal(loc = self._q_star[action], scale = np.sqrt(self.reward_variance))
        return reward
    
    @property
    def is_stationary(self) -> bool:
        return True