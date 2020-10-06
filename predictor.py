import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm


def train_predictor(diffusion, data_batches, lr=1e-4):
    predictor = None
    optim = None
    losses = []
    for batch in tqdm(data_batches):
        if predictor is None:
            predictor = Predictor(diffusion.num_steps, batch.shape[1:])
            optim = Adam(predictor.parameters(), lr=lr)
        ts = torch.randint(low=1, high=diffusion.num_steps + 1, size=(batch.shape[0],))
        epsilon = torch.randn(*batch.shape)
        samples = torch.from_numpy(
            diffusion.sample_q(batch, ts.numpy(), epsilon=epsilon.numpy())
        ).float()
        predictions = predictor(samples, ts)
        loss = torch.mean((epsilon - predictions) ** 2)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictor, losses


class Predictor(nn.Module):
    def __init__(self, num_steps, data_shape):
        super().__init__()
        self.num_steps = num_steps
        self.data_shape = data_shape

        self.timestep_coeff = torch.linspace(
            start=0.1 / num_steps, end=100 / num_steps, steps=128
        )[None]
        self.timestep_phase = nn.Parameter(torch.randn(128)[None])
        self.input_embed = nn.Linear(int(np.prod(data_shape)), 128)
        self.layers = nn.Sequential(
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, int(np.prod(data_shape))),
        )

    def forward(self, inputs, ts):
        embed_ts = torch.sin(
            (self.timestep_coeff * ts.float()[:, None]) + self.timestep_phase
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_ts)
        return out.view(inputs.shape)

    def predict_epsilon(self, inputs_np, ts_np):
        inputs = torch.from_numpy(inputs_np).float()
        ts = torch.from_numpy(ts_np).long()
        return self(inputs, ts).detach().numpy().astype(inputs_np.dtype)


class BayesPredictor(nn.Module):
    """
    An epsilon predictor that uses Bayes rule to predict epsilon without any
    learnable parameters--just a bunch of data.
    """

    def __init__(self, diffusion, data_batch):
        super().__init__()
        self.diffusion = diffusion
        self.data_batch = data_batch

    def forward(self, inputs, ts):
        alphas = self.diffusion.alphas_for_ts(ts, inputs.shape)
        means = np.sqrt(alphas)[:, None] * self.data_batch[None]
        variances = 1 - alphas
        while len(variances.shape) < len(means.shape):
            variances = variances[..., None]
        variances = torch.from_numpy(variances)
        means = torch.from_numpy(means)
        logits = -(
            0.5 * torch.log(variances)
            + (0.5 / variances) * (inputs[:, None] - means) ** 2
        )
        while len(logits.shape) > 2:
            logits = torch.sum(logits, dim=-1)
        logits -= torch.max(logits, dim=1, keepdims=True)[0]
        probs = torch.exp(logits)
        probs /= torch.sum(probs, dim=1, keepdims=True)
        while len(probs.shape) < len(self.data_batch.shape) + 1:
            probs = probs[..., None]
        x_0 = torch.sum(torch.from_numpy(self.data_batch[None]) * probs, dim=1)
        alphas = self.diffusion.alphas_for_ts(ts, inputs.shape)
        alphas = torch.from_numpy(alphas)
        return (inputs - torch.sqrt(alphas) * x_0) / torch.sqrt(1 - alphas)

    def predict_epsilon(self, inputs_np, ts_np):
        inputs = torch.from_numpy(inputs_np)
        ts = torch.from_numpy(ts_np)
        return self(inputs, ts).detach().numpy().astype(inputs_np.dtype)
