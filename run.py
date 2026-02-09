import sys
import argparse
import experiments.sutton_barto
import experiments.custom_stationary_env_high_variance
import experiments.utils

def main():
    parser = argparse.ArgumentParser(description="Run RL Modular Testbed Experiments")
    parser.add_argument('--experiment', type=str, default='Sutton and Barto All', 
                        help='Which experiment to run? (Options: Sutton and Barto All, ' \
                        'Exercise 2.5, Figure 2.2, Figure 2.3, Figure 2.4, Figure 2.5, Custom High Variance')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed logging')

    args = parser.parse_args()
    verbose = args.verbose
    experiment_type = args.experiment

    match experiment_type:
        case "Exercise 2.5":
            if verbose:
                print(">>> Starting Sutton & Barto Exercise 2.5...")
                experiments.utils.print_separator_line()

            experiments.sutton_barto.run_sutton_barto_exercise_2_5(book_figure_name="Exercise 2.5", 
                                                                   verbose=verbose)
            
            if verbose:
                experiments.utils.print_separator_line()

        case "Figure 2.2":
            if verbose:
                print(">>> Starting Sutton & Barto Figure 2.2...")
                experiments.utils.print_separator_line()

            experiments.sutton_barto.run_sutton_barto_figure_2_2(book_figure_name="Figure 2.2", 
                                                                 verbose=verbose)
            if verbose:
                experiments.utils.print_separator_line()

        case "Figure 2.3":
            if verbose:
                print(">>> Starting Sutton & Barto Figure 2.3...")
                experiments.utils.print_separator_line()

            experiments.sutton_barto.run_sutton_barto_figure_2_3(book_figure_name="Figure 2.3",
                                                                 verbose=verbose)
            
            if verbose:
                experiments.utils.print_separator_line()

        case "Figure 2.4":
            if verbose:
                print(">>> Starting Sutton & Barto Figure 2.4...")
                experiments.utils.print_separator_line()

            experiments.sutton_barto.run_sutton_barto_figure_2_4(book_figure_name="Figure 2.4",
                                                                verbose=verbose)
            if verbose:
                experiments.utils.print_separator_line()

        case "Figure 2.5":
            if verbose:
                print(">>> Starting Sutton & Barto Figure 2.5...")
                experiments.utils.print_separator_line()

            experiments.sutton_barto.run_sutton_barto_figure_2_5(book_figure_name="Figure 2.5", 
                                                                 verbose=verbose)
            
            if verbose:
                experiments.utils.print_separator_line()

        case "Sutton and Barto All":
            if verbose:
                print(">>> Starting Sutton & Barto Reproduction...")
            experiments.sutton_barto.run_all(verbose=verbose)

            if verbose:
                experiments.utils.print_separator_line()

        case "Custom High Variance":
            if verbose:
                print(">>> Starting Custom Experiment: High Variance Stress Test...")
                experiments.utils.print_separator_line()
                experiments.custom_stationary_env_high_variance.run_experiment(verbose=verbose)
            
            if verbose:
                experiments.utils.print_separator_line()

        case _:
            print("Unknown experiment.")
            raise NotImplementedError


if __name__ == "__main__":
    main()