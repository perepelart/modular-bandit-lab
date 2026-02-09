from testbeds.testbed import MultiArmedBanditTestbed
from plotting.learning_curve import LearningCurve
import configs.sutton_barto
import experiments.utils

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_sutton_barto_exercise_2_5(book_figure_name = None, verbose = False):
    """
    Reproduces Sutton & Barto Exercise 2.5 (p. 33).
    
    Investigates the limitations of sample-average methods in non-stationary 
    environments compared to constant step-size methods. The environment 
    drifts using a Gaussian random walk.
    """

    config = configs.sutton_barto.config_sample_averages_vs_constant_alpha_in_non_stationary_env

    n_runs = configs.sutton_barto.global_sim_config.N_RUNS
    t_steps = configs.sutton_barto.global_sim_config.T_STEPS_NON_STATIONARY

    testbed_nonstationary_example = MultiArmedBanditTestbed(**config)
    testbed_result = testbed_nonstationary_example.execute(n_runs = n_runs, t_steps = t_steps, verbose = verbose)

    plot_name = "Testing Sample Average step-size versus Constant in the Non-stationary Environment"
    plot_name = experiments.utils.add_book_figure_name(book_figure_name, plot_name)
    
    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot("Mean Reward", verbose = verbose)
    learning_curve.plot("Action Optimality", verbose = verbose)

    return testbed_result
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_sutton_barto_figure_2_2(book_figure_name = None, verbose = False):
    """
    Reproduces Sutton & Barto Figure 2.2 (p. 29).
    
    Compares the performance of different epsilon-greedy strategies (ε=0, 
    ε=0.01, ε=0.1) using sample-average step sizes in a stationary 
    Gaussian bandit environment.
    """
     
    config = configs.sutton_barto.config_stationary_env_exploration_vs_exploitation_or_balance

    n_runs = configs.sutton_barto.global_sim_config.N_RUNS
    t_steps = configs.sutton_barto.global_sim_config.T_STEPS_STATIONARY

    testbed_stationary_example = MultiArmedBanditTestbed(**config)
    testbed_result = testbed_stationary_example.execute(n_runs = n_runs, t_steps = t_steps, verbose = verbose)

    plot_name = "Testing Different epsilon-greedy Algorithms with Sample Average step-size and Stationary Environment"
    plot_name = experiments.utils.add_book_figure_name(book_figure_name, plot_name)

    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot("Mean Reward", verbose = verbose)
    learning_curve.plot("Action Optimality", verbose = verbose)

    return testbed_result
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_sutton_barto_figure_2_3(book_figure_name = None, verbose = False):
    """
    Reproduces Sutton & Barto Figure 2.3 (p. 34).
    
    Demonstrates the effect of Optimistic Initial Values (Q1=5) versus 
    Realistic Initial Values (Q1=0) on the exploration-exploitation 
    behavior of a greedy agent.
    """
    config = configs.sutton_barto.config_optimistic_initial_values_vs_realistic
    n_runs = configs.sutton_barto.global_sim_config.N_RUNS
    t_steps = configs.sutton_barto.global_sim_config.T_STEPS_STATIONARY


    testbed_optimistic_vs_realistic = MultiArmedBanditTestbed(**config)
    testbed_result = testbed_optimistic_vs_realistic.execute(n_runs = n_runs, t_steps = t_steps, verbose = verbose)

    plot_name = "Testing Optimistic Initial Values for Epsilon-Greedy Policy"
    plot_name = experiments.utils.add_book_figure_name(book_figure_name, plot_name)

    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot("Action Optimality", verbose = verbose)
    learning_curve.plot("Mean Reward", verbose = verbose)

    return testbed_result
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_sutton_barto_figure_2_4(book_figure_name = None, verbose = False):
    """
    Reproduces Sutton & Barto Figure 2.4 (p. 36).

    Compares Upper-Confidence-Bound (UCB, c=2) action selection against 
    a standard epsilon-greedy agent (ε=0.1) in a stationary environment.
    """

    config = configs.sutton_barto.config_ucb_vs_epsilon_greedy_stationary_env_sample_average_step_size
    n_runs = configs.sutton_barto.global_sim_config.N_RUNS
    t_steps = configs.sutton_barto.global_sim_config.T_STEPS_STATIONARY

    testbed_ucb_vs_epsilon_greedy = MultiArmedBanditTestbed(**config)
    testbed_result = testbed_ucb_vs_epsilon_greedy.execute(n_runs = n_runs, t_steps = t_steps, verbose = verbose)

    plot_name = "Upper-Confidence-Bound Action Selection Test"
    plot_name = experiments.utils.add_book_figure_name(book_figure_name, plot_name)

    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot('Mean Reward', verbose = verbose)

    return testbed_result
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_sutton_barto_figure_2_5(book_figure_name = None, verbose = False):
    """
    Reproduces Sutton & Barto Figure 2.5 (p. 38).
    
    Analyzes Gradient Bandit Algorithms, comparing performance with and 
    without a reward baseline across different step sizes (alpha).
    """

    config = configs.sutton_barto.config_gradient_agents_with_and_without_baselines
    n_runs = configs.sutton_barto.global_sim_config.N_RUNS
    t_steps = configs.sutton_barto.global_sim_config.T_STEPS_STATIONARY

    testbed_gradient_agents_with_and_without_baselines = MultiArmedBanditTestbed(**config)
    testbed_result = testbed_gradient_agents_with_and_without_baselines.execute(n_runs = n_runs, t_steps = t_steps, verbose = verbose)

    plot_name = "Gradient Bandit Algorithms Comparison"
    plot_name = experiments.utils.add_book_figure_name(book_figure_name, plot_name)

    learning_curve = LearningCurve(testbed_result, plot_name = plot_name)
    learning_curve.plot("Action Optimality", verbose = verbose)

    return testbed_result

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def run_all(verbose = False):
    if verbose:
        print()
        print(">>> Starting Sutton & Barto Exercise 2.5...")
        experiments.utils.print_separator_line()

    run_sutton_barto_exercise_2_5(book_figure_name="Exercise 2.5", verbose=verbose)

    if verbose:
        experiments.utils.print_separator_line()
        print()

    if verbose:
        print(">>> Starting Sutton & Barto Figure 2.2...")
        experiments.utils.print_separator_line()

    run_sutton_barto_figure_2_2(book_figure_name="Figure 2.2", verbose=verbose)

    if verbose:
        experiments.utils.print_separator_line()
        print()

    if verbose:
        print(">>> Starting Sutton & Barto Figure 2.3...")
        experiments.utils.print_separator_line()

    run_sutton_barto_figure_2_3(book_figure_name="Figure 2.3", verbose=verbose)

    if verbose:
        experiments.utils.print_separator_line()
        print()

    if verbose:
        print(">>> Starting Sutton & Barto Figure 2.4...")
        experiments.utils.print_separator_line()
    
    run_sutton_barto_figure_2_4(book_figure_name="Figure 2.4", verbose=verbose)
    
    if verbose:
        experiments.utils.print_separator_line()
        print()

    if verbose:
        print(">>> Starting Sutton & Barto Figure 2.5...")
        experiments.utils.print_separator_line()

    run_sutton_barto_figure_2_5(book_figure_name="Figure 2.5", verbose=verbose)

    if verbose:
        experiments.utils.print_separator_line()
        print()
        print("The reproduction completed.")


    