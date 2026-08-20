"""Physics-informed MLP-GRU predictor for future geometric entanglement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class StaticEntanglementPINN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class EntanglementMLPGRU(nn.Module):
    def __init__(self, feature_dim: int, hamiltonian_dim: int, output_steps: int = 19) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1)
        )
        self.gru = nn.GRU(64, 256, num_layers=2, dropout=0.1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(256 + hamiltonian_dim, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, output_steps),
        )

    def forward(self, sequence: torch.Tensor, hamiltonian: torch.Tensor) -> torch.Tensor:
        _sequence, hidden = self.gru(self.point_encoder(sequence))
        return self.decoder(torch.cat([hidden[-1], hamiltonian], dim=-1))


def entanglement_loss(prediction: torch.Tensor, target: torch.Tensor, trajectory: bool) -> torch.Tensor:
    data_loss = F.mse_loss(prediction, target)
    range_loss = torch.mean(F.relu(-prediction) ** 2 + F.relu(prediction - 1.0) ** 2)
    total = data_loss + 0.05 * range_loss
    if trajectory:
        slope = F.mse_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
        curve = F.mse_loss(
            prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2],
            target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2],
        )
        total = total + 0.20 * slope + 0.05 * curve
    return total


def load_data(data_root: Path, n: int, static: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = data_root / "prepared_features" / f"N{n}"
    cache = np.load(directory / "trajectory_feature_cache.npz")
    observations = cache["observations"].astype(np.float32)
    target = cache["eg"].astype(np.float32)
    hamiltonian = cache["hamiltonian"].astype(np.float32)
    times = cache["times"].astype(np.float32)
    metadata = pd.read_csv(directory / "trajectory_feature_cache_meta.csv")
    split_table = pd.read_csv(directory / "aligned_trajectory_split.csv")
    split_map = split_table.set_index("trajectory_id")["split"]
    split_names = metadata["trajectory_id"].astype(str).map(split_map).to_numpy()
    indices = {name: np.flatnonzero(split_names == name) for name in ("train", "validation", "test")}
    if static:
        return {"features": observations[:, 0], "target": target[:, 0, None]}, indices
    observed_steps = 12
    j_mean = hamiltonian[:, 0]
    tau = j_mean[:, None] * times[None, :observed_steps]
    normalized_time = np.broadcast_to(times[None, :observed_steps] / times[-1], tau.shape)
    time_encoding = np.stack([normalized_time, tau, np.sin(tau), np.cos(tau)], axis=-1)
    return {
        "sequence": np.concatenate([observations[:, :observed_steps], time_encoding], axis=-1).astype(np.float32),
        "hamiltonian": hamiltonian[:, 1:],
        "target": target[:, observed_steps:],
    }, indices


def scale(values: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    if values.ndim == 3:
        shape = values.shape
        scaler.fit(values[train_indices].reshape(-1, shape[-1]))
        return scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    scaler.fit(values[train_indices])
    return scaler.transform(values).astype(np.float32)


def make_loader(arrays: list[np.ndarray], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(*(torch.from_numpy(array[indices]) for array in arrays)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arrays, indices = load_data(args.data, args.n, args.static)
    trajectory = not args.static
    if args.static:
        features = scale(arrays["features"], indices["train"])
        model = StaticEntanglementPINN(features.shape[-1])
        model_arrays = [features, arrays["target"]]
    else:
        sequence = scale(arrays["sequence"], indices["train"])
        hamiltonian = scale(arrays["hamiltonian"], indices["train"])
        model = EntanglementMLPGRU(sequence.shape[-1], hamiltonian.shape[-1], arrays["target"].shape[-1])
        model_arrays = [sequence, hamiltonian, arrays["target"]]
    loaders = {
        name: make_loader(model_arrays, index, args.batch_size, name == "train")
        for name, index in indices.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_validation, stale = None, float("inf"), 0
    for _epoch in range(args.epochs):
        model.train()
        for batch in loaders["train"]:
            inputs, target = batch[:-1], batch[-1].to(device)
            loss = entanglement_loss(model(*(value.to(device) for value in inputs)), target, trajectory)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loaders["validation"]:
                inputs, target = batch[:-1], batch[-1].to(device)
                losses.append(float(entanglement_loss(model(*(value.to(device) for value in inputs)), target, trajectory)))
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
        for batch in loaders["test"]:
            inputs, target = batch[:-1], batch[-1]
            prediction.append(model(*(value.to(device) for value in inputs)).cpu().numpy())
            truth.append(target.numpy())
    truth, prediction = np.concatenate(truth).ravel(), np.concatenate(prediction).ravel()
    print({"mse": mean_squared_error(truth, prediction), "mae": mean_absolute_error(truth, prediction), "r2": r2_score(truth, prediction)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../../GME-data"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=Path("entanglement_mlp_gru.pt"))
    train(parser.parse_args())


if __name__ == "__main__":
    main()

