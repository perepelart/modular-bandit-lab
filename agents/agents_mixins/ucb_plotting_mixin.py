from agents.agents_mixins.q_value_agent_plotting_mixin import QValuePlottingMixin


class UCBPlottingMixin(QValuePlottingMixin):
    """Specific Mixin for UCBAgent."""
# --- Implementation of the required for plotting methods ---

    def get_trace_name(self, plotly = True):
        if not plotly:
            raise NotImplementedError
        
        c = self.exploration_degree
        agent_trace_name = f'<b>UCB</b>, c = {c}'    

        return agent_trace_name

    def get_description(self, plotly = True):
        if not plotly:
            raise NotImplementedError
        
        agent_description_strings = []

        policy_name_string = fr"\text{{Agent follows UCB policy}},"
        agent_description_strings.append(policy_name_string)

        exploration_degree_string = fr'c = {self.exploration_degree},'
        agent_description_strings.append(exploration_degree_string)

        q_value_init_value_string = super()._get_q_value_init_string()
        agent_description_strings.append(q_value_init_value_string)
        step_size_choice_string = super()._get_step_size_choice_string()
        agent_description_strings.append(step_size_choice_string)

        agent_compiled_description = r"$" + r" ".join(agent_description_strings) + r"$"
        return agent_compiled_description
