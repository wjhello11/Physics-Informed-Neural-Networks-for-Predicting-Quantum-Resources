"""Bayes-by-Backprop extension and uncertainty-based trajectory analysis.

This compact file contains the BayesianLinear layer, variational objective,
posterior sampling, interval calibration, and the ranking quantities reported
in the manuscript. Detailed plotting and experiment-management code is omitted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from PINN_entanglement_MLP_GRU import load_data, make_loader, scale


Z90 = 1.6448536269514722


class BayesianLinear(nn.Module):
    def __init__(self, inputs: int, outputs: int, prior_sigma: float = 1.0) -> None:
        super().__init__()
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs))
        self.weight_rho = nn.Parameter(torch.full((outputs, inputs), -5.5))
        self.bias_mu = nn.Parameter(torch.empty(outputs))
        self.bias_rho = nn.Parameter(torch.full((outputs,), -5.5))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.uniform_(self.bias_mu, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs))

    @staticmethod
    def sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-8

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight = self.weight_mu + self.sigma(self.weight_rho) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + self.sigma(self.bias_rho) * torch.randn_like(self.bias_mu)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(inputs, weight, bias)

    def kl(self) -> torch.Tensor:
        prior_variance = self.prior_sigma**2
        total = torch.zeros((), device=self.weight_mu.device)
        for mean, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            standard_deviation = self.sigma(rho)
            total = total + (
                math.log(self.prior_sigma) - torch.log(standard_deviation)
                + (standard_deviation.square() + mean.square()) / (2 * prior_variance) - 0.5
            ).sum()
        return total


class BayesianMLPGRU(nn.Module):
    def __init__(self, feature_dim: int, hamiltonian_dim: int, output_steps: int = 19) -> None:
        super().__init__()
        self.point = BayesianLinear(feature_dim, 64)
        self.norm = nn.LayerNorm(64)
        self.gru = nn.GRU(64, 256, num_layers=2, dropout=0.1, batch_first=True)
        self.decoder1 = BayesianLinear(256 + hamiltonian_dim, 256)
        self.decoder2 = BayesianLinear(256, 128)
        self.decoder3 = BayesianLinear(128, output_steps)
        self.log_noise_variance = nn.Parameter(torch.full((output_steps,), math.log(0.02**2)))

    def forward(self, sequence: torch.Tensor, hamiltonian: torch.Tensor, sample: bool = True) -> torch.Tensor:
        encoded = F.gelu(self.norm(self.point(sequence, sample)))
        _sequence, hidden = self.gru(encoded)
        values = F.silu(self.decoder1(torch.cat([hidden[-1], hamiltonian], dim=-1), sample))
        values = F.silu(self.decoder2(values, sample))
        return self.decoder3(values, sample)

    def kl(self) -> torch.Tensor:
        return self.point.kl() + self.decoder1.kl() + self.decoder2.kl() + self.decoder3.kl()


def variational_loss(
    model: BayesianMLPGRU,
    prediction: torch.Tensor,
    target: torch.Tensor,
    training_count: int,
    beta: float,
) -> torch.Tensor:
    variance = torch.exp(model.log_noise_variance).clamp_min(1e-8)
    nll = 0.5 * torch.mean((prediction - target).square() / variance + torch.log(variance))
    slope = F.mse_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    curve = F.mse_loss(
        prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2],
        target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2],
    )
    value_range = torch.mean(F.relu(-prediction) ** 2 + F.relu(prediction - 1) ** 2)
    return nll + beta * model.kl() / training_count + 0.20 * slope + 0.05 * curve + 0.05 * value_range


def posterior_samples(
    model: BayesianMLPGRU,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws, truth = [], []
    model.eval()
    with torch.no_grad():
        for sample_index in range(samples):
            rows, current_truth = [], []
            for sequence, hamiltonian, target in data_loader:
                rows.append(model(sequence.to(device), hamiltonian.to(device), sample=True).cpu().numpy())
                if sample_index == 0:
                    current_truth.append(target.numpy())
            draws.append(np.concatenate(rows))
            if sample_index == 0:
                truth = current_truth
    draws = np.asarray(draws)
    return draws.mean(axis=0), draws.std(axis=0, ddof=1), np.concatenate(truth)


def calibration_multiplier(truth: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray) -> float:
    standardized_error = np.abs(truth - mean) / np.maximum(standard_deviation, 1e-8)
    return float(np.quantile(standardized_error, 0.90) / Z90)


def uncertainty_summary(
    truth: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    multiplier: float,
) -> dict[str, float]:
    calibrated = standard_deviation * multiplier
    coverage = np.mean(np.abs(truth - mean) <= Z90 * calibrated)
    correlation = np.corrcoef(calibrated.ravel(), np.abs(truth - mean).ravel())[0, 1]
    return {"coverage_90": float(coverage), "uncertainty_error_correlation": float(correlation)}


def ranking_summary(truth: np.ndarray, mean: np.ndarray) -> pd.DataFrame:
    true_peak, predicted_peak = truth.max(axis=1), mean.max(axis=1)
    predicted_peak_time = mean.argmax(axis=1)
    usefulness = truth[np.arange(len(truth)), predicted_peak_time] / np.maximum(true_peak, 1e-8)
    rows = []
    for top_fraction in (0.10, 0.20, 0.30):
        count = max(1, int(np.ceil(top_fraction * len(truth))))
        selected = np.argsort(predicted_peak)[-count:]
        reference = np.argsort(true_peak)[-count:]
        overlap = len(np.intersect1d(selected, reference))
        recall = overlap / count
        rows.append(
            {
                "top_ranked_fraction": top_fraction,
                "recall": recall,
                "enrichment_factor": recall / top_fraction,
                "model_selected_mean_true_peak": float(true_peak[selected].mean()),
                "reference_top_mean_true_peak": float(true_peak[reference].mean()),
                "mean_peak_time_usefulness": float(usefulness[selected].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../../GME-data"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("bayesian_results"))
    args = parser.parse_args()

    arrays, indices = load_data(args.data, args.n, static=False)
    sequence = scale(arrays["sequence"], indices["train"])
    hamiltonian = scale(arrays["hamiltonian"], indices["train"])
    loaders = {
        name: make_loader([sequence, hamiltonian, arrays["target"]], index, 512, name == "train")
        for name, index in indices.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianMLPGRU(sequence.shape[-1], hamiltonian.shape[-1], arrays["target"].shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _epoch in range(args.epochs):
        model.train()
        for seq, ham, target in loaders["train"]:
            prediction = model(seq.to(device), ham.to(device), sample=True)
            loss = variational_loss(model, prediction, target.to(device), len(indices["train"]), beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    validation_mean, validation_std, validation_truth = posterior_samples(model, loaders["validation"], device, args.samples)
    multiplier = calibration_multiplier(validation_truth, validation_mean, validation_std)
    test_mean, test_std, test_truth = posterior_samples(model, loaders["test"], device, args.samples)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"posterior_N{args.n}.npz", mean=test_mean, std=test_std, truth=test_truth)
    pd.DataFrame([uncertainty_summary(test_truth, test_mean, test_std, multiplier)]).to_csv(
        args.output / f"uncertainty_N{args.n}.csv", index=False
    )
    ranking_summary(test_truth, test_mean).to_csv(args.output / f"ranking_N{args.n}.csv", index=False)
    torch.save(model.state_dict(), args.output / f"bayesian_model_N{args.n}.pt")


if __name__ == "__main__":
    main()

