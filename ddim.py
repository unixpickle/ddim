import numpy as np


def create_alpha_schedule(num_steps=100, beta_0=0.0001, beta_T=0.02):
    betas = np.linspace(beta_0, beta_T, num_steps)
    result = [1.0]
    alpha = 1.0
    for beta in betas:
        alpha *= 1 - beta
        result.append(alpha)
    return np.array(result, dtype=np.float64)


class Diffusion:
    """
    A numpy implementation of the DDPM and DDIM setup.
    """

    def __init__(self, alphas):
        self.alphas = alphas

    @property
    def num_steps(self):
        return len(self.alphas) - 1

    def sample_q(self, x_0, ts, epsilon=None):
        """
        Sample from q(x_t | x_0) for a batch of x_0.
        """
        if epsilon is None:
            epsilon = np.random.normal(size=x_0.shape)
        alphas = self.alphas_for_ts(ts, x_0.shape)
        return np.sqrt(alphas) * x_0 + np.sqrt(1 - alphas) * epsilon

    def ddim_previous(self, x_t, ts, epsilon_prediction):
        """
        Take a ddim sampling step given x_t, t, and epsilon prediction.
        """
        alphas = self.alphas_for_ts(ts, x_t.shape)
        x_0 = (x_t - np.sqrt(1 - alphas) * epsilon_prediction) / np.sqrt(alphas)
        return self.sample_q(x_0, ts - 1, epsilon=epsilon_prediction)

    def ddim_sample(self, x_T, predictor):
        """
        Sample x_0 from x_t using DDIM, assuming a method on predictor called
        predict_epsilon(x_t, ts).
        """
        x_t = x_T
        for t in range(1, self.num_steps)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            x_t = self.ddim_previous(x_t, ts, predictor.predict_epsilon(x_t, ts))
        return x_t

    def ddim_sample_cond(self, x_T, predictor, x_cond, mask):
        """
        Like ddim_sample(), but condition on the part of x_cond which is True
        in the boolean mask.
        """
        x_t = x_T
        for t in range(1, self.num_steps)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            x_t = np.where(mask, self.sample_q(x_cond, ts), x_t)
            x_t = self.ddim_previous(x_t, ts, predictor.predict_epsilon(x_t, ts))
        return np.where(mask, x_cond, x_t)

    def ddpm_previous(self, x_t, ts, epsilon_prediction, epsilon=None):
        if epsilon is None:
            epsilon = np.random.normal(size=x_t.shape)
        alphas_t = self.alphas_for_ts(ts, x_t.shape)
        alphas_prev = self.alphas_for_ts(ts - 1, x_t.shape)
        alphas = alphas_t / alphas_prev
        betas = 1 - alphas
        return (1 / np.sqrt(alphas)) * (
            x_t - betas / np.sqrt(1 - alphas_t) * epsilon_prediction
        ) + np.sqrt(betas) * epsilon

    def ddpm_sample(self, x_T, predictor):
        """
        Sample x_0 from x_t using DDPM.

        Usage is the same as ddim_sample().
        """
        x_t = x_T
        for t in range(1, self.num_steps)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            x_t = self.ddpm_previous(x_t, ts, predictor.predict_epsilon(x_t, ts))
        return x_t

    def ddpm_sample_cond(self, x_T, predictor, x_cond, mask, num_subsamples=1):
        """
        Create a masked-conditional sample using DDPM.

        See ddim_sample_cond() for usage details.
        """
        x_t = x_T
        for t in range(1, self.num_steps)[::-1]:
            samples = []
            for _ in range(num_subsamples):
                ts = np.array([t] * x_T.shape[0])
                x_t = np.where(mask, self.sample_q(x_cond, ts), x_t)
                x_next = self.ddpm_previous(x_t, ts, predictor.predict_epsilon(x_t, ts))
                samples.append(x_next)
            x_t = np.mean(samples, axis=0)
        return np.where(mask, x_cond, x_t)

    def alphas_for_ts(self, ts, shape):
        alphas = self.alphas[ts]
        while len(alphas.shape) < len(shape):
            alphas = alphas[..., None]
        return alphas
