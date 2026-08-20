"""Core state, SDP-label, dynamics, and noise code for E_G experiments.

Full precomputed labels are distributed in GME-data.zip. The implementation
below shows the numerical definitions used in the manuscript; checkpointing,
parallel cloud execution, and plotting utilities are intentionally omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)
FAMILY_RATIOS = {"randpure_fsep_mix": 0.80, "haar_random_pure": 0.10, "fully_separable_k8": 0.10}


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def haar_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return state / np.linalg.norm(state)


def product_state(n: int, rng: np.random.Generator) -> np.ndarray:
    return kron_all([haar_state(2, rng) for _ in range(n)])


def separable_k8(n: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet(np.ones(8))
    return sum(float(w) * projector(product_state(n, rng)) for w in weights)


def generate_state(family: str, n: int, rng: np.random.Generator, mix_max: float = 1.0) -> np.ndarray:
    if family == "haar_random_pure":
        return projector(haar_state(2**n, rng))
    if family == "fully_separable_k8":
        return separable_k8(n, rng)
    if family == "randpure_fsep_mix":
        mixture = float(rng.uniform(0.0, mix_max))
        return (1 - mixture) * projector(haar_state(2**n, rng)) + mixture * separable_k8(n, rng)
    raise ValueError(f"Unknown family: {family}")


def random_xy_hamiltonian(n: int, couplings: np.ndarray, fields: np.ndarray) -> np.ndarray:
    hamiltonian = np.zeros((2**n, 2**n), dtype=np.complex128)
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


def bits(index: int, n: int) -> list[int]:
    return [(index >> (n - 1 - site)) & 1 for site in range(n)]


def bit_index(values: list[int]) -> int:
    result = 0
    for value in values:
        result = (result << 1) | value
    return result


def partial_transpose_map(n: int, subsystem: tuple[int, ...]) -> sparse.csr_matrix:
    dim = 2**n
    rows, columns = [], []
    for row in range(dim):
        for column in range(dim):
            row_bits, column_bits = bits(row, n), bits(column, n)
            for site in subsystem:
                row_bits[site], column_bits[site] = column_bits[site], row_bits[site]
            rows.append(row * dim + column)
            columns.append(bit_index(row_bits) * dim + bit_index(column_bits))
    return sparse.csr_matrix((np.ones(dim * dim), (rows, columns)), shape=(dim * dim, dim * dim))


def geometric_entanglement(rho: np.ndarray, n: int, solver: str = "MOSEK") -> float:
    """PPT-relaxed SDP label used as E_G throughout the released experiments."""
    dim = 2**n
    sigma = cp.Variable((dim, dim), hermitian=True)
    auxiliary = cp.Variable((dim, dim), complex=True)
    constraints = [sigma >> 0, cp.trace(sigma) == 1]
    constraints.append(cp.bmat([[cp.Constant(rho), auxiliary], [auxiliary.H, sigma]]) >> 0)
    for site in range(n):
        permutation = partial_transpose_map(n, (site,))
        transposed = cp.reshape(permutation @ cp.vec(sigma, order="C"), (dim, dim), order="C")
        constraints.append(transposed >> 0)
    problem = cp.Problem(cp.Maximize(cp.real(cp.trace(auxiliary))), constraints)
    value = problem.solve(solver=solver, verbose=False)
    fidelity = float(np.clip(np.real(value), 0.0, 1.0))
    return float(np.clip(1.0 - fidelity**2, 0.0, 1.0))


def reduced_state(rho: np.ndarray, n: int, keep: int) -> np.ndarray:
    tensor = rho.reshape([2] * (2 * n))
    current_n = n
    for site in sorted(set(range(n)) - {keep}, reverse=True):
        tensor = np.trace(tensor, axis1=site, axis2=site + current_n)
        current_n -= 1
    return tensor.reshape(2, 2)


def trace_power(rho: np.ndarray, power: int) -> float:
    return float(np.trace(np.linalg.matrix_power(rho, power)).real)


def low_order_features(rho: np.ndarray, n: int) -> np.ndarray:
    diagonal = np.diag(rho).real
    local_moments, polarizations = [], []
    for site in range(n):
        local = reduced_state(rho, n, site)
        local_moments.extend([trace_power(local, 2), trace_power(local, 3)])
        polarizations.append(abs(float((local[0, 0] - local[1, 1]).real)))
    nonzero = diagonal[diagonal > 1e-12]
    diagonal_entropy = -float(np.sum(nonzero * np.log2(nonzero))) / np.log2(2**n)
    structural = [float(diagonal.max()), diagonal_entropy, np.mean(polarizations), np.min(polarizations)]
    return np.asarray(
        [*diagonal, trace_power(rho, 2), trace_power(rho, 3), *local_moments, *structural],
        dtype=np.float32,
    )


def evolve(rho0: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, time: float) -> np.ndarray:
    phases = np.exp(-1j * eigenvalues * time)
    unitary = (eigenvectors * phases[None, :]) @ eigenvectors.conj().T
    return unitary @ rho0 @ unitary.conj().T


def depolarize(rho: np.ndarray, probability: float) -> np.ndarray:
    return (1 - probability) * rho + probability * np.eye(len(rho)) / len(rho)


def family_sequence(total: int) -> list[str]:
    counts = {name: int(total * ratio) for name, ratio in FAMILY_RATIOS.items()}
    counts["randpure_fsep_mix"] += total - sum(counts.values())
    return [name for name, count in counts.items() for _ in range(count)]


def generate(n: int, trajectories: int, output: Path, seed: int, solver: str) -> None:
    rng = np.random.default_rng(seed + n)
    times = np.linspace(0.0, 3.0, 31, dtype=np.float32)
    states, features, labels, hamiltonians = [], [], [], []
    families = family_sequence(trajectories)
    rng.shuffle(families)
    mix_max = 0.8 if n == 3 else 1.0
    for family in families:
        rho0 = generate_state(family, n, rng, mix_max=mix_max)
        couplings = rng.uniform(0.8, 1.2, n - 1)
        fields = rng.uniform(-0.2, 0.2, n)
        eigenvalues, eigenvectors = np.linalg.eigh(random_xy_hamiltonian(n, couplings, fields))
        trajectory_features, trajectory_labels = [], []
        for time in times:
            rho_t = evolve(rho0, eigenvalues, eigenvectors, float(time))
            trajectory_features.append(low_order_features(rho_t, n))
            trajectory_labels.append(geometric_entanglement(rho_t, n, solver=solver))
        states.append(rho0)
        features.append(trajectory_features)
        labels.append(trajectory_labels)
        hamiltonians.append([float(np.mean(couplings)), *couplings, *fields])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"entanglement_N{n}.npz",
        rho0=np.asarray(states), observations=np.asarray(features, dtype=np.float32),
        eg=np.asarray(labels, dtype=np.float32), hamiltonian=np.asarray(hamiltonians, dtype=np.float32),
        times=times, family=np.asarray(families),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../../GME-data/generated"))
    parser.add_argument("--n-list", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    for n in args.n_list:
        generate(n, args.trajectories, args.output, args.seed, args.solver)


if __name__ == "__main__":
    main()

