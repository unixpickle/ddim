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

        self.timestep_embed = nn.Embedding(num_steps + 1, 128)
        self.input_embed = nn.Linear(int(np.prod(data_shape)), 128)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, int(np.prod(data_shape))),
        )

    def forward(self, inputs, ts):
        embed_ts = self.timestep_embed(ts)
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_ts)
        return out.view(inputs.shape)

    def predict_epsilon(self, inputs_np, ts_np):
        inputs = torch.from_numpy(inputs_np).float()
        ts = torch.from_numpy(ts_np).long()
        return self(inputs, ts).detach().numpy().astype(inputs_np.dtype)
