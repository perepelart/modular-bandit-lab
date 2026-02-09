class StationaryGaussianBanditMixin:
    def get_description(self, plotly = True):    
        if not plotly:
            raise NotImplementedError
        
        reward_variance_value = self.reward_variance

        init_mean_value = self.env_init_mean
        init_variance_value = self.env_init_variance

        math_segments = []

        math_segments.append(fr"\textbf{{Information about the environment ({self.n_arms}-armed bandit problem):}}")
        math_segments.append(fr"\text{{Expected reward }} \mathbb{{E}}[R_t | A_t = a]=q_*(a) \text{{ was sampled from }} \mathcal{{N}}({init_mean_value},{init_variance_value}) \text{{ for each action }} a.")

        math_segments.append(fr"R_t | A_t = a \sim \mathcal{{N}}(q_*(a), {reward_variance_value})")

        math_segments.append(fr"\text{{The environment is stationary.}}")

        environment_description_compiled = r"$" + r" \\ ".join(math_segments) + r"$"
        return environment_description_compiled