from agents.constants import StepSizeMode

class QValuePlottingMixin:
    """
    BASE Mixin for Q-value based agents. Generates common description parts.
    """
    def _get_q_value_init_string(self):
        if self.optimistic_initialization:
            q_value_init_value = self.optimistic_constant
        else:
            q_value_init_value = 0

        q_value_init_value_string = fr'Q_0 = {q_value_init_value},'        
        
        return q_value_init_value_string
    
    def _get_step_size_choice_string(self):
        step_size_mode = self.step_size_mode
        match step_size_mode:
            case StepSizeMode.SAMPLE_AVERAGE:
                step_size_choice_string = fr'Q_{{n+1}} \gets Q_n + \frac{{1}}{{n}}[R_n - Q_n]'
            case StepSizeMode.CONSTANT:
                step_size = self.step_size
                step_size_choice_string = fr'Q_{{n+1}} \gets Q_n + {step_size}[R_n - Q_n]'
            case StepSizeMode.UNBIASED_TRICK:
                step_size_choice_string = fr'Q_{{n+1}} \gets Q_n + \alpha_n[R_n - Q_n]'
            case _:
                raise NotImplementedError
            
        return step_size_choice_string