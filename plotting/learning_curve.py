import os

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from agents.constants import StepSizeMode, BaselineMode
from testbeds.testbed_result import TestbedResult

class LearningCurve:
    """
    Handles the generation of Plotly visualizations for Testbed results.
    
    Supports 'Mean Reward' and 'Action Optimality' plots with standard 
    deviation shading and MathJax-rendered descriptions.
    """
    def __init__(self, testbed_result: TestbedResult, plot_name):
            self.figure = go.Figure()
            self.plot_width = 0.9
            self.agents_description = []
            self.testbed_result = testbed_result
            self.colors_rgba = [
                    'rgba(31, 119, 180, {alpha})',   # Muted Blue
                    'rgba(255, 127, 14, {alpha})',   # Safety Orange
                    'rgba(44, 160, 44, {alpha})',    # Cooked Asparagus Green
                    'rgba(214, 39, 40, {alpha})',    # Brick Red
                    'rgba(148, 103, 189, {alpha})',  # Muted Purple
                ]
            self.plot_name = plot_name

            if not os.path.exists("images"):
                os.makedirs("images")

    def plot(self, statistics, verbose = False):
        if verbose:
            print(f"Plotting {statistics} in progress...")

        self.figure = go.Figure()
        for agent_number in range(self.testbed_result.n_agents):
            self.__plot_mean_and_std_lines(agent_number, statistics)
            

        self.__place_annotation()
        self.__update_layout(statistics)
        self.figure.show()

        match statistics:
            case "Action Optimality":
                self.figure.write_html(f"images/action_optimality_plot_{self.plot_name}.html")

                file_name = self.plot_name.replace(" ", "_").replace(":", "_")
                pio.write_image(self.figure, f"images/action_optimality_plot_{file_name}.png", width=1584, height=836, scale=2)
            case "Mean Reward":
                self.figure.write_html(f"images/mean_rewards_plot_{self.plot_name}.html")

                file_name = self.plot_name.replace(" ", "_").replace(":", "_")
                pio.write_image(self.figure, f"images/mean_rewards_plot_{file_name}.png", width=1584, height=836, scale=2)

        if verbose:
            print(f"Plotting {statistics} completed.\n")

    def __plot_mean_and_std_lines(self, agent_number, statistics):
        '''
        self.rewards_history_per_run.shape = (number_of_agents, N, T)
        Hence self.rewards_history_per_run[agent_number] has shape (N, T)

        R_11 R_21 ... R_1T
        R_21 R_22 ... R_2T
        ...  ...  ... ...
        R_N1 R_N2 ... R_NT


        So np.mean(self.rewards_history_per_run[agent_number], axis = 0) calculates mean for each column (changing row)
        and we get the rewards averaged for every 1...T epoch by all runs 1...N.
        '''

        match statistics:
            case "Action Optimality":
                mean = np.mean(self.testbed_result.raw_action_optimality[agent_number]*100, axis = 0)
                std = np.std(self.testbed_result.raw_action_optimality[agent_number]*100, axis = 0)
                trace_name = "% Optimal Action"
            case "Mean Reward":
                mean = np.mean(self.testbed_result.raw_rewards[agent_number], axis = 0)
                std = np.std(self.testbed_result.raw_rewards[agent_number], axis = 0)
                trace_name = "Mean Reward"

        epochs_x = np.arange(self.testbed_result.timesteps) # X-axis for this agent

        color_template = self.colors_rgba[agent_number % len(self.colors_rgba)] # Cycle through defined colors


        # --- Add Standard Deviation Band ---
        # Upper bound trace (invisible line)
        self.figure.add_trace(go.Scatter(
            x = epochs_x,
            y = mean + std,
            mode = 'lines',
            line=dict(width=0), # Make line invisible
            showlegend=False,   # Don't show this trace in legend
            hoverinfo='none'    # No hover text for bounds
        ))

        # Lower bound trace (invisible line, fills to the upper bound)
        self.figure.add_trace(go.Scatter(
            x = epochs_x,
            y = mean - std,
            mode = 'lines',
            line=dict(width=0),
            fillcolor=color_template.format(alpha=0.2), # Fill color with transparency
            fill='tonexty', # Fill area between this trace and the previous one (upper bound)
            showlegend=False,
            hoverinfo='none'
        ))


        agent_trace_name = self.testbed_result.agent_traces_names[agent_number]
        # --- Add Mean Reward Line ---
        # Added last so it appears on top of the shaded area
        self.figure.add_trace(go.Scatter(
            x = epochs_x,
            y = mean,
            mode = 'lines',
            name = f'{self.testbed_result.agents_descriptions[agent_number]} | {trace_name}',
            line=dict(color=color_template.format(alpha=1.0)), # Solid line color
            hovertemplate = agent_trace_name + f'<br>Epoch: | %{{x}}<br>{trace_name}: %{{y:.4f}}<extra></extra>' # Custom hover text
        ))

    
    def __place_annotation(self):
        annotation_text = self.testbed_result.environment_information

        # --- Add Annotation for Environment Information ---
        self.figure.add_annotation(
            text = annotation_text,
            align = 'left',
            showarrow = False,
            xref = 'paper',
            yref = 'paper',
            x = self.plot_width + 0.01,
            y = 0.6, # Start high for a tall block of text
            xanchor = "left",
            yanchor = "top", # Anchor top of the text block
            font = dict(size=12),
            bordercolor = 'black', # Changed color for new test
            borderwidth = 1,
            bgcolor = 'rgba(0,0,0,0.0)', # white
            # No explicit width/height initially, let the single MathJax block size itself
        )
    
    
    def __update_layout(self, statistics):

        match statistics:
            case "Action Optimality":
                set_title = f'Agent Performance Comparison (Averaged Optimal Action over {self.testbed_result.n_runs} runs)'
                set_yaxis_title = '% Optimal Action'
                set_yaxis_ticksuffix = '%'
                set_yaxis_range = [0, 100]
            case "Mean Reward":
                set_title = f'Agent Performance Comparison (Mean Reward over {self.testbed_result.n_runs} runs)'
                set_yaxis_title = 'Averaged by Run Test Reward per Epoch'
                set_yaxis_ticksuffix = None
                set_yaxis_range = self.__calculate_yaxis_range()
            case _:
                raise NotImplementedError


        self.figure.update_layout(
        title = self.plot_name +': '+ set_title,
        xaxis=dict(
                domain=[0, self.plot_width],  # Plot area will only use 0% to 70% of the width
                title='Epochs (Time steps)'
            ),
        xaxis_title ='Epochs (Time steps)',
        yaxis_title = set_yaxis_title,
        yaxis_ticksuffix = set_yaxis_ticksuffix,
        hovermode = "x unified", # Show hover info for all traces at a given x-coordinate
        legend_title_text = f'Agent',
        legend=dict(
            traceorder = "normal", # Order legend items as they were added (Mean lines first conceptually)
            # Adjust legend position if needed:
            x = self.plot_width + 0.01,           # Start at 72% of the width (in the new panel)
            y=1,              # Place at the top
            xanchor='left',   # Anchor the legend's left side
            yanchor='top'     # Anchor the legend's top side
            ),
        paper_bgcolor='white',
        legend_bgcolor = 'white',
        yaxis_range = set_yaxis_range,
        margin=dict(r=450)
        )

    def __calculate_yaxis_range(self, upper_padding_factor = 1.1, lower_padding_factor_positive = 0.9, lower_padding_factor_negative = 1.1):
        '''
        Calculates the y-axis range for plotting rewards.

        The upper bound is based on the maximum of (mean + std dev) across all agents and epochs.
        The lower bound is based on the minimum of (mean - std dev) across all agents and epochs.
        Padding is applied to these bounds.

        Args:
            upper_padding_factor (float): Factor to extend the upper bound (e.g., 1.1 for 10% padding).
            lower_padding_factor_positive (float): Factor to adjust the lower bound if it's positive
                                                (e.g., 0.9 to move it 10% towards zero).
            lower_padding_factor_negative (float): Factor to adjust the lower bound if it's negative
                                                (e.g., 1.1 to move it 10% further from zero).
        Returns:
            list: A list containing two elements [y_min, y_max].
        '''
        if self.testbed_result.n_agents == 0 or self.testbed_result.raw_rewards.size == 0:
            # Default range if no agents or no data is present
            return [0.0, 1.0]

        all_agents_max_envelope_points = []
        all_agents_min_envelope_points = []

        for agent_idx in range(self.testbed_result.n_agents):
            # Extract data for the current agent; shape should be (N, T)
            agent_rewards = self.testbed_result.raw_rewards[agent_idx]

            # Skip if this agent has no reward data (e.g., N or T is 0 for this agent's data)
            if agent_rewards.ndim != 2 or agent_rewards.shape[0] == 0 or agent_rewards.shape[1] == 0:
                # Assuming N (runs) is axis 0 and T (epochs) is axis 1 for agent_rewards
                # If agent_rewards is not 2D or one of the dimensions is zero.
                # Based on the problem statement, agent_rewards should have shape (N,T)
                # So np.mean(agent_rewards, axis=0) expects axis 0 to be N.
                continue

            # Mean rewards over N runs for each of T epochs
            mean_over_runs = np.mean(agent_rewards, axis=0)  # Shape (T,)
            # Standard deviation of rewards over N runs for each of T epochs
            std_over_runs = np.std(agent_rewards, axis=0)    # Shape (T,)

            # Calculate the highest point of the (mean + std) envelope for this agent
            max_epoch_value = np.max(mean_over_runs + std_over_runs)
            all_agents_max_envelope_points.append(max_epoch_value)

            # Calculate the lowest point of the (mean - std) envelope for this agent
            min_epoch_value = np.min(mean_over_runs - std_over_runs)
            all_agents_min_envelope_points.append(min_epoch_value)

        if not all_agents_max_envelope_points and not all_agents_min_envelope_points:
            # This case occurs if all agents had empty or invalid data
            return [0.0, 1.0] # Fallback default range

        # Determine overall maximum and minimum values from the envelopes across all agents
        # If a list is empty, np.max/min would raise an error.
        overall_max_val = np.max(all_agents_max_envelope_points) if all_agents_max_envelope_points else 0.0
        overall_min_val = np.min(all_agents_min_envelope_points) if all_agents_min_envelope_points else 0.0

        # Calculate final y-axis bounds with padding
        y_max_final = overall_max_val * upper_padding_factor

        if overall_min_val >= 0:
            y_min_final = overall_min_val * lower_padding_factor_positive
        else: # overall_min_val is negative
            y_min_final = overall_min_val * lower_padding_factor_negative
        
        # Handle cases where calculated min and max are too close or identical (e.g., all data is zero)
        if np.isclose(y_min_final, y_max_final):
            if np.isclose(y_min_final, 0.0):  # If range is close to [0,0]
                y_min_final = -0.1
                y_max_final = 0.1
            else: # If range is close but non-zero, e.g. [5,5]
                center_val = y_min_final # (y_min_final + y_max_final) / 2.0
                spread = abs(center_val * 0.1) # 10% of the value
                if np.isclose(spread, 0.0): # If center_val is tiny, use absolute spread
                    spread = 0.1 
                y_min_final = center_val - spread
                y_max_final = center_val + spread
        
        # Ensure y_min is strictly less than y_max
        if y_min_final >= y_max_final:
            # This is a fallback if previous logic still resulted in min >= max
            # (e.g. if initial overall_min_val and overall_max_val were identical and non-zero,
            #  and padding factors were such that y_min_final ended up >= y_max_final,
            #  or if the isclose block for non-zero values didn't create enough separation)
            temp_avg = (overall_min_val + overall_max_val) / 2.0
            temp_span = abs(overall_max_val - overall_min_val)
            if temp_span < 1e-6: # If original data was essentially flat
                temp_span = abs(temp_avg * 0.2) if abs(temp_avg * 0.2) > 1e-5 else 0.2
            
            y_min_final = temp_avg - (temp_span / 2.0) * max(lower_padding_factor_negative, upper_padding_factor) # Use larger factor for safety
            y_max_final = temp_avg + (temp_span / 2.0) * max(lower_padding_factor_negative, upper_padding_factor)

            if np.isclose(y_min_final, y_max_final): # Ultimate fallback for zero data
                y_min_final = -0.1
                y_max_final = 0.1


        return [y_min_final, y_max_final]

