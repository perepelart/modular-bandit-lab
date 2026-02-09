from dataclasses import dataclass
import numpy as np


@dataclass
class TestbedResult:
    """
    A self-contained, serializable record of a Testbed run.
    """
    raw_rewards: np.ndarray  # Shape: (num_agents, num_runs, num_steps)
    raw_action_optimality: np.ndarray

    agents_descriptions: list[str] 
    agent_traces_names : list[str] 
    environment_information: str
    n_runs: int
    timesteps: int
    
    @property
    def n_agents(self) -> int:
        return len(self.agents_descriptions)