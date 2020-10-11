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
            predictor = Predictor(batch.shape[1:])
            optim = Adam(predictor.parameters(), lr=lr)
        ts = torch.randint(low=1, high=diffusion.num_steps + 1, size=(batch.shape[0],))
        epsilon = torch.randn(*batch.shape)
        samples = torch.from_numpy(
            diffusion.sample_q(batch, ts.numpy(), epsilon=epsilon.numpy())
        ).float()
        alphas = torch.from_numpy(diffusion.alphas_for_ts(ts.numpy()))
        predictions = predictor(samples, alphas.float())
        loss = torch.mean((epsilon - predictions) ** 2)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictor, losses


class Predictor(nn.Module):
    def __init__(self, data_shape, num_layers=1, channels=128):
        super().__init__()
        self.data_shape = data_shape

        self.timestep_coeff = torch.linspace(start=0.1, end=100, steps=channels)[None]
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(data_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(data_shape))),
        )

    def forward(self, inputs, alphas):
        embed_alphas = torch.sin(
            (self.timestep_coeff * alphas.float()[:, None]) + self.timestep_phase
        )
        embed_alphas = self.timestep_embed(embed_alphas)
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_alphas)
        return out.view(inputs.shape)

    def predict_epsilon(self, inputs_np, alphas_np):
        inputs = torch.from_numpy(inputs_np).float()
        alphas = torch.from_numpy(alphas_np).float()
        return self(inputs, alphas).detach().numpy().astype(inputs_np.dtype)


class BayesPredictor(nn.Module):
    """
    An epsilon predictor that uses Bayes rule to predict epsilon without any
    learnable parameters--just a bunch of data.
    """

    def __init__(self, data_batch):
        super().__init__()
        self.data_batch = data_batch

    def forward(self, inputs, alphas):
        while len(alphas.shape) < len(inputs.shape):
            alphas = alphas[..., None]
        means = torch.sqrt(alphas)[:, None] * self.data_batch[None]
        variances = 1 - alphas
        while len(variances.shape) < len(means.shape):
            variances = variances[..., None]
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
        return (inputs - torch.sqrt(alphas) * x_0) / torch.sqrt(1 - alphas)

    def predict_epsilon(self, inputs_np, alphas_np):
        inputs = torch.from_numpy(inputs_np)
        alphas = torch.from_numpy(alphas_np)
        return self(inputs, alphas).detach().numpy().astype(inputs_np.dtype)
