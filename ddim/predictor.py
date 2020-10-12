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
        dev = next(predictor.parameters()).device
        ts = torch.randint(
            low=1, high=diffusion.num_steps + 1, size=(batch.shape[0],)
        ).to(dev)
        epsilon = torch.randn(*batch.shape).to(dev)
        samples = (
            torch.from_numpy(
                diffusion.sample_q(
                    batch, ts.cpu().numpy(), epsilon=epsilon.cpu().numpy()
                )
            )
            .float()
            .to(dev)
        )
        alphas = torch.from_numpy(diffusion.alphas_for_ts(ts.cpu().numpy())).to(dev)
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

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
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
        dev = next(self.parameters()).device
        inputs = torch.from_numpy(inputs_np).float().to(dev)
        alphas = torch.from_numpy(alphas_np).float().to(dev)
        with torch.no_grad():
            return self(inputs, alphas).detach().cpu().numpy().astype(inputs_np.dtype)


class CNNPredictor(nn.Module):
    def __init__(self, data_shape, num_res_blocks=5, channels=128):
        super().__init__()
        assert len(data_shape) == 3
        self.data_shape = data_shape

        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=1000, steps=channels * 4)[None],
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels * 4)[None])
        self.timestep_embed = nn.Sequential(
            nn.Linear(channels * 4, channels), nn.GELU(), nn.Linear(channels, channels),
        )
        self.input_embed = nn.Conv2d(data_shape[0], channels, 1)
        self.res_blocks = nn.ModuleList([])
        for i in range(num_res_blocks):
            block = nn.Sequential(
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                SELayer(channels),
            )
            self.res_blocks.append(block)
        self.out_layer = nn.Conv2d(channels, data_shape[0], 3, padding=1)

    def forward(self, inputs, alphas):
        assert inputs.shape[1:] == self.data_shape
        embed_alphas = torch.sin(
            (self.timestep_coeff * alphas.float()[:, None]) + self.timestep_phase
        )
        embed_alphas = self.timestep_embed(embed_alphas)[..., None, None]
        out = self.input_embed(inputs)
        for block in self.res_blocks:
            out = out + block(out + embed_alphas)
        out = self.out_layer(out)
        return out

    def predict_epsilon(self, inputs_np, alphas_np):
        dev = next(self.parameters()).device
        inputs = torch.from_numpy(inputs_np).float().to(dev)
        alphas = torch.from_numpy(alphas_np).float().to(dev)
        with torch.no_grad():
            return self(inputs, alphas).detach().cpu().numpy().astype(inputs_np.dtype)


class SELayer(nn.Module):
    """
    https://github.com/moskomule/senet.pytorch/blob/23839e07525f9f5d39982140fccc8b925fe4dee9/senet/se_module.py
    """

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
