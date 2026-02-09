class DriftingGaussianBanditMixin:
    def get_description(self, plotly = True):
            if not plotly:
                raise NotImplementedError

            reward_variance_value = self.reward_variance
            drift_mean_value = self.drift_mean
            drit_variance_value = self.drift_variance

            math_segments = []

            math_segments.append(fr"\textbf{{Information about the environment (Non-stationary {self.n_arms}-armed bandit problem):}}")
            math_segments.append(fr"\text{{Expected reward }} \mathbb{{E}}[R_t | A_t = a]=q_*(a) \text{{ was initialized with zeros}}\text{{ for each action }} a.")
            math_segments.append(fr"R_t | A_t = a \sim \mathcal{{N}}(q_*(a), {reward_variance_value})")

            math_segments.append(
                        fr"\text{{The environment is not stationary and its mean follows a Gaussian random walk, }}"
                    )
            math_segments.append(fr"q_*^t(a) =  q_*^{{t-1}}(a) + Z_t, ~ Z_t \sim \mathcal{{N}}({drift_mean_value}, {drit_variance_value})")

                        # Join all segments with MathJax newline '\\' and wrap in a SINGLE $...$
            environment_description_compiled = r"$" + r" \\ ".join(math_segments) + r"$"

            return environment_description_compiled
