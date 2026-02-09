import numpy as np
from tqdm import tqdm

from testbeds.testbed_result import TestbedResult

class MultiArmedBanditTestbed:
    """
    Orchestrates the interaction between Agents and Environments.
    
    Manages the experiment loop, data collection, and random seeding 
    to ensure reproducibility.

    environment_prototype: dict -- config of the environment
    agent_prototypes: list -- list of configs of agents


    random_seed: int -- to make results of the experiments deterministic between runs of the program
    """

    def __init__(self, 
                 environment_prototype: dict,
                 agent_prototypes: list,

                 random_seed = 2048):
        
        self.environment_prototype = environment_prototype
        self.environment = environment_prototype['class'](**environment_prototype['params'])
        self.agent_prototypes = agent_prototypes
        self.n_agents = len(self.agent_prototypes)

        self.random_seed = random_seed
        np.random.seed(random_seed)




    #one run of T time steps of each algorithm
    def __run(self, agents, t_steps):
        self.environment.reset()

        for agent_number in range(self.n_agents):
            agents[agent_number].initialize(self.environment)

        agents_rewards_history = np.zeros((self.n_agents, t_steps + 1))
        agents_action_optimality_history = np.zeros((self.n_agents, t_steps + 1))

        for t in range(1, t_steps + 1):
            current_true_optimal_action_for_env = self.environment.optimal_action

            # --- Each agent interacts with the environment ---
            for agent_number in range(self.n_agents):
                # 1. Agent chooses an action
                action_taken_by_agent = agents[agent_number].select_action()

                # 2. Check if this agent's action was truly optimal
                if action_taken_by_agent == current_true_optimal_action_for_env:
                    agents_action_optimality_history[agent_number][t] = 1

                # 3. Sample reward for the agent based on its chosen action
                # Assuming self.reward_variance is a scalar (same variance for all arms)
                # If arm-specific variance: self.reward_variances_per_arm[action_taken_by_agent]                
                reward_for_agent = self.environment.sample_reward(action_taken_by_agent)
                agents_rewards_history[agent_number][t] = reward_for_agent
                agents[agent_number].update_policy(action_taken_by_agent, reward_for_agent, t)

            #if the environment is non-stationary, we need to be carefully to use the same distribution for each agent at time t to sample rewards
            # --- Non-Stationary Update for the shared environment (once per time step) ---
            if not self.environment.is_stationary:
                self.environment.step()


        
        return agents_rewards_history, agents_action_optimality_history

    '''
    n_runs: int -- number of experiments per t_step
    t_steps: int -- number of epochs (time steps)    
    '''
    def execute(self, n_runs: int = 2000, t_steps: int = 1000, verbose = False):
        self.t_steps = t_steps
        self.n_runs = n_runs

        rewards_history_per_run = np.empty(shape = (self.n_agents, n_runs, t_steps + 1))
        action_optimality_history_per_run = np.empty(shape = (self.n_agents, n_runs, t_steps + 1))
        agents = [prototype['class'](**prototype['params']) for prototype in self.agent_prototypes]

        if verbose:
            print(f"Starting {n_runs} runs of {t_steps} timesteps...\n")

        for i in tqdm(range(n_runs), disable = not verbose):

            agents_rewards_history, agents_action_optimalily_history = self.__run(agents, t_steps)

            for agent_number in range(self.n_agents):
                rewards_history_per_run[agent_number][i] = agents_rewards_history[agent_number]
                action_optimality_history_per_run[agent_number][i] = agents_action_optimalily_history[agent_number]

                agents[agent_number] = agents[agent_number].clone()


        if verbose:
            print(f"Completed {n_runs} runs of {t_steps} timesteps.\n")

        agent_traces_names = [agent.get_trace_name() for agent in agents]
        agent_descriptions = [agent.get_description() for agent in agents]
        environment_description = self.environment.get_description()
        return TestbedResult(agents_descriptions = agent_descriptions,
                             agent_traces_names = agent_traces_names,
                             environment_information = environment_description,
                             timesteps = t_steps,
                             n_runs = self.n_runs,
                             raw_rewards = rewards_history_per_run,
                             raw_action_optimality = action_optimality_history_per_run)