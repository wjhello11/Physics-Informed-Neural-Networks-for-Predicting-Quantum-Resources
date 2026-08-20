"""Core data-generation code for the coherence experiments.

The released archive contains the full N=3,4,5 datasets. This file keeps the
state families, observable features, random-coupling XY evolution, and labels
needed to regenerate them without exposing unrelated engineering utilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_RATIOS = {"pure_mixture": 0.60, "haar_pure": 0.20, "pure_diag_mixture": 0.20}
MIXTURE_COMPONENTS = {3: 6, 4: 35, 5: 50}
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def random_xy_hamiltonian(n: int, couplings: np.ndarray, fields: np.ndarray) -> np.ndarray:
    dim = 2**n
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for site, coupling in enumerate(couplings):
        for operator in (X, Y):
            ops = [I2] * n
            ops[site], ops[site + 1] = operator, operator
            hamiltonian += float(coupling) * kron_all(ops)
    for site, field in enumerate(fields):
        ops = [I2] * n
        ops[site] = Z
        hamiltonian += float(field) * kron_all(ops)
    return (hamiltonian + hamiltonian.conj().T) / 2


def pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return state / np.linalg.norm(state)


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def random_pure_mixture(dim: int, components: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet(np.ones(components))
    return sum(float(w) * projector(pure_state(dim, rng)) for w in weights)


def generate_state(family: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    dim = 2**n
    if family == "haar_pure":
        return projector(pure_state(dim, rng)), 0.0
    if family == "pure_mixture":
        return random_pure_mixture(dim, MIXTURE_COMPONENTS[n], rng), 0.0
    if family == "pure_diag_mixture":
        mix = float(rng.random())
        probabilities = rng.dirichlet(np.ones(dim))
        rho = (1 - mix) * projector(pure_state(dim, rng)) + mix * np.diag(probabilities)
        return rho, mix
    raise ValueError(f"Unknown family: {family}")


def trace_power(rho: np.ndarray, power: int) -> float:
    return float(np.real(np.trace(np.linalg.matrix_power(rho, power))))


def entropy(rho: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh((rho + rho.conj().T) / 2).real, 0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def coherence_labels(rho: np.ndarray) -> np.ndarray:
    dim = rho.shape[0]
    l1_value = (np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
    relative_entropy = (entropy(np.diag(np.diag(rho))) - entropy(rho)) / np.log2(dim)
    return np.array([l1_value, relative_entropy], dtype=np.float32)


def z_string_features(rho: np.ndarray, n: int) -> np.ndarray:
    probabilities = np.diag(rho).real
    basis = np.arange(2**n)
    values = []
    for mask in range(2**n):
        parity = np.ones(2**n)
        for qubit in range(n):
            if mask & (1 << qubit):
                parity *= 1 - 2 * ((basis >> qubit) & 1)
        values.append(float(parity @ probabilities))
    return np.asarray(values, dtype=np.float32)


def input_features(
    rho0: np.ndarray,
    n: int,
    couplings: np.ndarray,
    fields: np.ndarray,
    time: float,
) -> np.ndarray:
    static = np.concatenate(
        [z_string_features(rho0, n), [trace_power(rho0, 2), trace_power(rho0, 3)]]
    )
    tau = float(np.mean(couplings) * time)
    return np.concatenate([static, couplings, fields, [tau, np.sin(tau), np.cos(tau)]]).astype(np.float32)


def evolve(rho0: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, time: float) -> np.ndarray:
    phases = np.exp(-1j * eigenvalues * time)
    unitary = (eigenvectors * phases[None, :]) @ eigenvectors.conj().T
    return unitary @ rho0 @ unitary.conj().T


def exact_family_counts(total: int) -> dict[str, int]:
    counts = {name: int(total * ratio) for name, ratio in FAMILY_RATIOS.items()}
    counts["pure_mixture"] += total - sum(counts.values())
    return counts


def generate_dataset(n: int, trajectories: int, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed + n)
    times = np.linspace(0.0, 3.0, 31, dtype=np.float32)
    x_rows, y_rows, metadata = [], [], []
    trajectory_index = 0
    for family, count in exact_family_counts(trajectories).items():
        for _ in range(count):
            rho0, mixture = generate_state(family, n, rng)
            couplings = rng.uniform(0.8, 1.2, n - 1)
            fields = rng.uniform(-0.2, 0.2, n)
            eigenvalues, eigenvectors = np.linalg.eigh(random_xy_hamiltonian(n, couplings, fields))
            trajectory_id = f"N{n}_{family}_{trajectory_index:05d}"
            for time_index, time in enumerate(times):
                rho_t = evolve(rho0, eigenvalues, eigenvectors, float(time))
                x_rows.append(input_features(rho0, n, couplings, fields, float(time)))
                y_rows.append(coherence_labels(rho_t))
                metadata.append(
                    {"trajectory_id": trajectory_id, "family": family, "time_index": time_index,
                     "t": float(time), "mixture_t": mixture}
                )
            trajectory_index += 1
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"svr_coherence_random_xy_N{n}.npz",
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        time_grid=times,
    )
    pd.DataFrame(metadata).to_csv(output / f"svr_coherence_random_xy_N{n}_metadata.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../../coherence-data/datasets"))
    parser.add_argument("--n-list", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--trajectories", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    for n in args.n_list:
        generate_dataset(n, args.trajectories, args.output, args.seed)


if __name__ == "__main__":
    main()

