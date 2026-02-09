from testbeds.testbed import MultiArmedBanditTestbed
from plotting.learning_curve import LearningCurve
import configs.custom_stationary_env_high_variance

def run_experiment(verbose = False):
    """
    Executes the High Variance Stress Test.
    
    Compares the stability of UCB (c=2) vs Epsilon-Greedy (eps=0.1) 
    in an environment with 10x standard reward variance.
    """
    
    config = configs.custom_stationary_env_high_variance.config_high_variance_comparison
    settings = configs.custom_stationary_env_high_variance.custom_sim_config
    
    testbed = MultiArmedBanditTestbed(**config)
    testbed_result = testbed.execute(n_runs=settings.N_RUNS, t_steps = settings.T_STEPS, verbose = verbose)
    
    plot_name = "Custom Experiment: High Variance Stress Test (Reward Variance = 10)"
    
    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot("Mean Reward", verbose = verbose)
    learning_curve.plot("Action Optimality", verbose = verbose)

    return testbed_result