import numpy as np
import torch
from tqdm.auto import tqdm


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

    def predict_x0(self, x_t, ts, epsilon_prediction):
        alphas = self.alphas_for_ts(ts, x_t.shape)
        return (x_t - np.sqrt(1 - alphas) * epsilon_prediction) / np.sqrt(alphas)

    def ddim_previous(self, x_t, ts, epsilon_prediction):
        """
        Take a ddim sampling step given x_t, t, and epsilon prediction.
        """
        x_0 = self.predict_x0(x_t, ts, epsilon_prediction)
        return self.sample_q(x_0, ts - 1, epsilon=epsilon_prediction)

    def ddim_sample(self, x_T, predictor, progress=False):
        """
        Sample x_0 from x_t using DDIM, assuming a method on predictor called
        predict_epsilon(x_t, alphas).
        """
        x_t = x_T
        t_iter = range(1, self.num_steps + 1)[::-1]
        if progress:
            t_iter = tqdm(t_iter)
        for t in t_iter:
            ts = np.array([t] * x_T.shape[0])
            alphas = self.alphas_for_ts(ts)
            x_t = self.ddim_previous(x_t, ts, predictor.predict_epsilon(x_t, alphas))
        return x_t

    def ddim_sample_cond(self, x_T, predictor, x_cond, mask):
        """
        Like ddim_sample(), but condition on the part of x_cond which is True
        in the boolean mask.
        """
        x_t = x_T
        for t in range(1, self.num_steps + 1)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            alphas = self.alphas_for_ts(ts)
            x_t = np.where(mask, self.sample_q(x_cond, ts), x_t)
            x_t = self.ddim_previous(x_t, ts, predictor.predict_epsilon(x_t, alphas))
        return np.where(mask, x_cond, x_t)

    def ddpm_previous(
        self, x_t, ts, epsilon_prediction, epsilon=None, cond_prediction=None
    ):
        if epsilon is None:
            epsilon = np.random.normal(size=x_t.shape)
        alphas_t = self.alphas_for_ts(ts, x_t.shape)
        alphas_prev = self.alphas_for_ts(ts - 1, x_t.shape)
        alphas = alphas_t / alphas_prev
        betas = 1 - alphas
        prev_mean = (1 / np.sqrt(alphas)) * (
            x_t - betas / np.sqrt(1 - alphas_t) * epsilon_prediction
        )
        if cond_prediction is not None:
            prev_mean += betas * cond_prediction
        return prev_mean + np.sqrt(betas) * epsilon

    def ddpm_sample(self, x_T, predictor):
        """
        Sample x_0 from x_t using DDPM.

        Usage is the same as ddim_sample().
        """
        x_t = x_T
        for t in range(1, self.num_steps + 1)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            alphas = self.alphas_for_ts(ts)
            x_t = self.ddpm_previous(x_t, ts, predictor.predict_epsilon(x_t, alphas))
        return x_t

    def ddpm_sample_cond(self, x_T, predictor, x_cond, mask, num_subsamples=1):
        """
        Create a masked-conditional sample using DDPM.

        See ddim_sample_cond() for usage details.
        """
        x_t = x_T
        for t in range(1, self.num_steps + 1)[::-1]:
            samples = []
            for _ in range(num_subsamples):
                ts = np.array([t] * x_T.shape[0])
                x_t = np.where(mask, self.sample_q(x_cond, ts), x_t)
                alphas = self.alphas_for_ts(ts)
                x_next = self.ddpm_previous(
                    x_t, ts, predictor.predict_epsilon(x_t, alphas)
                )
                samples.append(x_next)
            x_t = np.mean(samples, axis=0)
        return np.where(mask, x_cond, x_t)

    def ddpm_sample_cond_energy(self, x_T, predictor, cond_fn):
        """
        Create a sample using an energy function cond_fn as a conditioning
        signal, to compute p(x)*p(y|x), where cond_fn is grad_x log(p(y|x)).
        """
        x_t = x_T
        for t in range(1, self.num_steps + 1)[::-1]:
            ts = np.array([t] * x_T.shape[0])
            alphas = self.alphas_for_ts(ts)
            x_t = self.ddpm_previous(
                x_t,
                ts,
                predictor.predict_epsilon(x_t, alphas),
                cond_prediction=cond_fn(x_t, alphas),
            )
        return x_t

    def ddpm_sample_cond_energy_inpaint(
        self, x_T, predictor, x_cond, mask, temp=1.0, eps=1e-2
    ):
        def cond_fn(x_t, alphas):
            while len(alphas.shape) < len(x_t.shape):
                alphas = alphas[..., None]
            with torch.enable_grad():
                alphas_torch = torch.from_numpy(alphas).float()
                x_t_torch = torch.from_numpy(x_t).float().requires_grad_(True)
                eps_pred = predictor(x_t_torch, alphas_torch.view(-1))
                x_start = (
                    x_t_torch - (1 - alphas_torch).sqrt() * eps_pred
                ) / alphas_torch.sqrt()

                # This should be the variance of the x_start prediction,
                # but instead we use the variance of a signal noised to
                # the current timestep as a reasonable guess.
                sigmas = eps + 1 - alphas_torch

                log_density = -((torch.from_numpy(x_cond) - x_start) ** 2) / (
                    2 * sigmas
                )
                loss = (log_density * torch.from_numpy(mask).float()).sum()
                grad = torch.autograd.grad(loss, x_t_torch)[0]
                return grad.detach().numpy() / temp

        return self.ddpm_sample_cond_energy(x_T, predictor, cond_fn)

    def alphas_for_ts(self, ts, shape=None):
        alphas = self.alphas[ts]
        if shape is None:
            return alphas
        while len(alphas.shape) < len(shape):
            alphas = alphas[..., None]
        return alphas
