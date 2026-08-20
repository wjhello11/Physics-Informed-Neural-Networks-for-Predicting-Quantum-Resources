"""Physics-informed pointwise predictor for coherence dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CoherencePINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def pinn_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    data_loss = F.mse_loss(prediction, target)
    range_loss = torch.mean(F.relu(-prediction) ** 2 + F.relu(prediction - 1.0) ** 2)
    return data_loss + 0.10 * range_loss


def split_trajectories(count: int, seed: int) -> dict[str, np.ndarray]:
    order = np.random.default_rng(seed).permutation(count)
    first, second = int(0.70 * count), int(0.85 * count)
    return {"train": order[:first], "validation": order[first:second], "test": order[second:]}


def load_data(path: Path, n: int, static: bool) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    archive = np.load(path / "datasets" / f"svr_coherence_random_xy_N{n}.npz")
    time_points = len(archive["time_grid"])
    trajectory_count = len(archive["x"]) // time_points
    features = archive["x"].reshape(trajectory_count, time_points, -1).astype(np.float32)
    labels = archive["y"].reshape(trajectory_count, time_points, 2).astype(np.float32)
    split = split_trajectories(trajectory_count, 20260716)
    if static:
        return features[:, 0, :(2**n + 2)], labels[:, 0], split
    row_split = {
        name: (indices[:, None] * time_points + np.arange(time_points)).ravel()
        for name, indices in split.items()
    }
    return features.reshape(-1, features.shape[-1]), labels.reshape(-1, 2), row_split


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(features[indices]), torch.from_numpy(labels[indices])),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    features, labels, split = load_data(args.data, args.n, args.static)
    scaler = StandardScaler().fit(features[split["train"]])
    features = scaler.transform(features).astype(np.float32)
    loaders = {
        name: make_loader(features, labels, indices, args.batch_size, name == "train")
        for name, indices in split.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoherencePINN(features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_validation, stale = None, float("inf"), 0
    for _epoch in range(args.epochs):
        model.train()
        for x_batch, y_batch in loaders["train"]:
            loss = pinn_loss(model(x_batch.to(device)), y_batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for x_batch, y_batch in loaders["validation"]:
                losses.append(float(pinn_loss(model(x_batch.to(device)), y_batch.to(device))))
        validation = float(np.mean(losses))
        if validation < best_validation:
            best_validation, stale = validation, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    truth, prediction = [], []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loaders["test"]:
            prediction.append(model(x_batch.to(device)).cpu().numpy())
            truth.append(y_batch.numpy())
    truth, prediction = np.concatenate(truth).ravel(), np.concatenate(prediction).ravel()
    print(
        {
            "mse": mean_squared_error(truth, prediction),
            "mae": mean_absolute_error(truth, prediction),
            "r2": r2_score(truth, prediction),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../../coherence-data"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=Path("coherence_pinn.pt"))
    train(parser.parse_args())


if __name__ == "__main__":
    main()

