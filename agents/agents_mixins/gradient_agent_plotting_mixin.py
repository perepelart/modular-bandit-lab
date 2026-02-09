from agents.constants import BaselineMode

class GradientAgentPlottingMixin:
    def get_trace_name(self, plotly = True):
        if not plotly:
            raise NotImplementedError
        
        step_size = self.step_size
        agent_trace_name = f'<b>Gradient Agent</b>, α = {step_size}'

        return agent_trace_name
    
    def get_description(self, plotly = True):
        if not plotly:
            raise NotImplementedError
        
        agent_description_strings = []

        policy_name_string = fr"\text{{Agent follows the gradient-based policy}},"
        agent_description_strings.append(policy_name_string)

        baseline_mode = self.baseline_mode
        step_size = self.step_size


        match baseline_mode:
           case BaselineMode.SAMPLE_AVERAGE:
              step_size_choice_string = fr'H_{{t+1}}(A_t) \gets H_{{t}}(A_t) + {step_size}(R_t - \bar{{R}}_t)(1-\pi_t(A_t))'
           case BaselineMode.CONSTANT:
              baseline_constant = self.baseline_value
              step_size_choice_string = fr'H_{{t+1}}(A_t) \gets H_{{t}}(A_t) + {step_size}(R_t - {baseline_constant})(1-\pi_t(A_t))'
           case _:
              raise NotImplementedError
            
        agent_description_strings.append(step_size_choice_string)

        agent_description = r"$" + r" ".join(agent_description_strings) + r"$"
        return agent_description