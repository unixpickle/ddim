import argparse
import os

import torch
from torchvision import datasets, transforms

from ddim import Diffusion, CNNPredictor, create_alpha_schedule

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cpu" if not USE_CUDA else "cuda")
SAVE_PATH = "mnist_model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--save-interval", default=1000, type=int)
    args = parser.parse_args()
    train_data, test_data = create_datasets(args.batch_size, USE_CUDA)
    diffusion = Diffusion(
        create_alpha_schedule(num_steps=100, beta_0=0.001, beta_T=0.2)
    )
    model = CNNPredictor((1, 28, 28))
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH))
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    loaders = zip(iterate_loader(train_data), iterate_loader(test_data))
    for train_batch, test_batch in loaders:
        train_loss = compute_loss(diffusion, model, train_batch)
        with torch.no_grad():
            test_loss = compute_loss(diffusion, model, test_batch)
        print(f"step {step}: test={test_loss:.5f} train={train_loss:.5f}")
        step += 1
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        if not step % args.save_interval:
            model.cpu()
            torch.save(model.state_dict(), SAVE_PATH)
            model.to(DEVICE)


def compute_loss(diffusion, model, batch):
    ts = torch.randint(low=1, high=diffusion.num_steps + 1, size=(batch.shape[0],)).to(
        batch.device
    )
    epsilon = torch.randn_like(batch)
    samples = (
        torch.from_numpy(
            diffusion.sample_q(
                batch.cpu().numpy(), ts.cpu().numpy(), epsilon=epsilon.cpu().numpy()
            )
        )
        .float()
        .to(batch.device)
    )
    alphas = torch.from_numpy(diffusion.alphas_for_ts(ts.cpu().numpy())).to(
        batch.device
    )
    predictions = model(samples, alphas.float())
    return torch.mean((epsilon - predictions) ** 2)


def iterate_loader(loader):
    while True:
        for x, _ in loader:
            yield x.to(DEVICE)


def create_datasets(batch, use_cuda):
    # Taken from pytorch MNIST demo.
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    main()
