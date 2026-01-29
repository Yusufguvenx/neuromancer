"""
pde_data.py
Data generation module for PDE Control with DeepONet.

This module provides:
1. Crank-Nicolson PDE solver for heat equation with time-varying control
2. Gaussian Random Field generation for smooth control sequences
3. Dataset creation and neuromancer DictDataset loaders
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

from neuromancer.dataset import DictDataset

from pde_config import (
    L, T_FINAL, D, BETA, ALPHA, INITIAL_STATE_VAL, V_REF_VAL,
    NX_SOLVER, NT_SOLVER, X_GRID_SOLVER, SENSOR_LOCATIONS, M_SENSORS,
    NUM_BASIS_FUNCTIONS, NUM_TRAIN_SIMULATIONS, NUM_TEST_SIMULATIONS,
    DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, SURROGATE_BATCH_SIZE
)


def solve_pde_time_varying(u_control_sequence):
    """
    Solves the heat equation using Crank-Nicolson for a time-varying control sequence.

    PDE: dT/dt = D * d^2T/dx^2 - BETA*(T - V_REF) + ALPHA*u(x,t)

    Args:
        u_control_sequence: numpy array of shape (NT_SOLVER, NX_SOLVER) containing
                           the control input u(x,t) at each grid point and time step.

    Returns:
        numpy array of shape (NT_SOLVER, NX_SOLVER) containing the solution T(x,t).
    """
    dx = L / (NX_SOLVER - 1)
    dt = T_FINAL / (NT_SOLVER - 1)
    V_current = np.full(NX_SOLVER, INITIAL_STATE_VAL)
    V_history = [V_current.copy()]

    lambda_ = D * dt / (2 * dx**2)
    beta_term = 0.5 * BETA * dt

    # Create sparse matrices for implicit Crank-Nicolson scheme
    A_main_diag_vals = np.full(NX_SOLVER, 1 + 2 * lambda_ + beta_term)
    A_off_diag_vals = np.full(NX_SOLVER, -lambda_)
    A_diagonals = [A_off_diag_vals, A_main_diag_vals, A_off_diag_vals]

    B_main_diag_vals = np.full(NX_SOLVER, 1 - 2 * lambda_ - beta_term)
    B_off_diag_vals = np.full(NX_SOLVER, lambda_)
    B_diagonals = [B_off_diag_vals, B_main_diag_vals, B_off_diag_vals]

    A = spdiags(A_diagonals, [-1, 0, 1], NX_SOLVER, NX_SOLVER, format='csc')
    B = spdiags(B_diagonals, [-1, 0, 1], NX_SOLVER, NX_SOLVER, format='csc')

    # Boundary conditions (Neumann - zero flux)
    A = A.tolil()
    B = B.tolil()
    A[0, 1], A[-1, -2] = -2 * lambda_, -2 * lambda_
    B[0, 1], B[-1, -2] = 2 * lambda_, 2 * lambda_
    A = A.tocsc()
    B = B.tocsc()

    for k in range(NT_SOLVER - 1):
        # Average the control input over the timestep for Crank-Nicolson stability
        avg_u_in_step = (u_control_sequence[k] + u_control_sequence[k+1]) / 2.0
        source_term = ALPHA * avg_u_in_step + BETA * V_REF_VAL

        b_vec = B @ V_current + source_term * dt
        V_next = spsolve(A, b_vec)
        V_current = V_next
        V_history.append(V_current.copy())

    return np.array(V_history)


def generate_grf_time_series(num_steps, num_series, length_scale):
    """
    Generates smooth 1D Gaussian Random Field time series.

    Args:
        num_steps: Number of time steps
        num_series: Number of independent series (e.g., basis function weights)
        length_scale: Correlation length scale (larger = smoother)

    Returns:
        numpy array of shape (num_steps, num_series)
    """
    t = np.linspace(0, T_FINAL, num_steps)
    dist_matrix = np.abs(t[:, None] - t[None, :])
    cov_matrix = np.exp(-0.5 * (dist_matrix**2) / (length_scale**2)) + 1e-6 * np.eye(num_steps)
    return np.random.multivariate_normal(np.zeros(num_steps), cov_matrix, size=num_series).T


def create_dataset(num_simulations, filename):
    """
    Generates and saves a dataset of PDE trajectories.

    The dataset contains:
    - control_sequences: shape (N_sims, NT_SOLVER, NUM_BASIS_FUNCTIONS) - basis function weights
    - state_sequences: shape (N_sims, NT_SOLVER, M_SENSORS) - state at sensor locations

    Args:
        num_simulations: Number of simulation trajectories to generate
        filename: Path to save the .npz file
    """
    print(f"--- Generating Dataset: {filename} ---")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create basis functions matrix: cos(i*pi*x/L) for i=0,...,NUM_BASIS_FUNCTIONS-1
    basis_functions = np.cos(np.arange(NUM_BASIS_FUNCTIONS) * np.pi * X_GRID_SOLVER[:, None] / L)

    control_sequences = []
    state_sequences_at_sensors = []

    for i in range(num_simulations):
        if (i + 1) % 50 == 0:
            print(f"  Generating simulation {i+1}/{num_simulations}...")

        # 1. Generate diverse time-varying control weights using GRF
        length_scale = np.random.uniform(0.8, 2.5)  # Randomize smoothness
        w_sequence = generate_grf_time_series(NT_SOLVER, NUM_BASIS_FUNCTIONS, length_scale)
        w_sequence = np.clip(w_sequence * 0.7, -1.0, 1.0)  # Scale and clip weights

        # 2. Reconstruct full u(x,t) field from weights
        u_xt_sequence = w_sequence @ basis_functions.T  # (NT_SOLVER, NX_SOLVER)

        # 3. Solve PDE to get ground truth state evolution
        V_xt_solution = solve_pde_time_varying(u_xt_sequence)

        # 4. Interpolate state to sensor locations (if sensors != grid)
        interpolator = interp1d(X_GRID_SOLVER, V_xt_solution, axis=1,
                               kind='cubic', fill_value="extrapolate")
        V_at_sensors = interpolator(SENSOR_LOCATIONS.cpu().numpy().flatten())

        control_sequences.append(w_sequence)
        state_sequences_at_sensors.append(V_at_sensors)

    np.savez_compressed(
        filename,
        control_sequences=np.array(control_sequences),  # (N_sims, NT_SOLVER, NUM_BASIS_FUNCTIONS)
        state_sequences=np.array(state_sequences_at_sensors)  # (N_sims, NT_SOLVER, M_SENSORS)
    )
    print(f"Dataset saved to {filename}")


def get_surrogate_dataloaders(batch_size=None):
    """
    Returns neuromancer DictDatasets and DataLoaders for surrogate (DeepONet) training.

    The surrogate learns one-step transitions: (T_k, u_k) -> T_{k+1}

    Returns:
        train_loader, test_loader: PyTorch DataLoaders with neuromancer DictDataset
    """
    if batch_size is None:
        batch_size = SURROGATE_BATCH_SIZE

    # Generate data if not exists
    if not os.path.exists(TRAIN_DATA_PATH):
        create_dataset(NUM_TRAIN_SIMULATIONS, TRAIN_DATA_PATH)
    if not os.path.exists(TEST_DATA_PATH):
        create_dataset(NUM_TEST_SIMULATIONS, TEST_DATA_PATH)

    # Load data
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)

    # Create basis functions at sensor locations for reconstructing u(x) from weights
    sensor_x = SENSOR_LOCATIONS.cpu().numpy().flatten()
    basis_at_sensors = np.cos(np.arange(NUM_BASIS_FUNCTIONS) * np.pi * sensor_x[:, None] / L)
    basis_at_sensors_T = torch.from_numpy(basis_at_sensors.T).float()  # (NUM_BASIS_FUNCTIONS, M_SENSORS)

    def process_data(data):
        """Convert trajectory data to one-step samples."""
        weights = torch.from_numpy(data['control_sequences'][:, :-1, :]).float()
        weights = weights.reshape(-1, NUM_BASIS_FUNCTIONS)  # (N_samples, NUM_BASIS_FUNCTIONS)

        # Reconstruct control at sensor locations: u_k = weights @ basis^T
        control = weights @ basis_at_sensors_T  # (N_samples, M_SENSORS)

        state_inputs = torch.from_numpy(data['state_sequences'][:, :-1, :]).float()
        state_inputs = state_inputs.reshape(-1, M_SENSORS)  # T_k

        state_outputs = torch.from_numpy(data['state_sequences'][:, 1:, :]).float()
        state_outputs = state_outputs.reshape(-1, M_SENSORS)  # T_{k+1}

        return {
            'T_k': state_inputs,
            'u_k': control,
            'T_target': state_outputs,
            'x_locs': SENSOR_LOCATIONS.unsqueeze(0).expand(state_inputs.shape[0], -1, -1)
        }

    train_dict = process_data(train_data)
    test_dict = process_data(test_data)

    train_dataset = DictDataset(train_dict, name='train')
    test_dataset = DictDataset(test_dict, name='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             collate_fn=train_dataset.collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            collate_fn=test_dataset.collate_fn, shuffle=False)

    return train_loader, test_loader


def get_controller_dataloaders(batch_size=None):
    """
    Returns data needed for controller training.

    For controller training, we don't need pre-generated trajectories since
    we do self-supervised training with differentiable rollouts.
    This function returns the basis functions and initial state info.

    Returns:
        dict with 'basis_at_sensors' tensor and 'sensor_locations' tensor
    """
    sensor_x = SENSOR_LOCATIONS.cpu().numpy().flatten()
    basis_at_sensors = np.cos(np.arange(NUM_BASIS_FUNCTIONS) * np.pi * sensor_x[:, None] / L)

    return {
        'basis_at_sensors_T': torch.from_numpy(basis_at_sensors.T).float(),  # (NUM_BASIS_FUNCTIONS, M_SENSORS)
        'sensor_locations': SENSOR_LOCATIONS,  # (M_SENSORS, 1)
        'trunk_inputs': SENSOR_LOCATIONS.unsqueeze(0)  # (1, M_SENSORS, 1)
    }


if __name__ == "__main__":
    print("Generating training and test datasets...")
    print(f"Device: {torch.cuda.is_available()}")

    # Generate datasets
    create_dataset(NUM_TRAIN_SIMULATIONS, TRAIN_DATA_PATH)
    create_dataset(NUM_TEST_SIMULATIONS, TEST_DATA_PATH)

    # Test loading
    train_loader, test_loader = get_surrogate_dataloaders()
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Check shapes
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape}")
