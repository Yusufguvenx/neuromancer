"""
pde_evaluate.py
Evaluation and visualization of trained surrogate and controller models.

This script provides:
1. Surrogate evaluation: Multi-step rollout comparison with ground truth PDE solver
2. Controller evaluation: Closed-loop control performance on various target profiles
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from pde_config import (
    DEVICE, M_SENSORS, NUM_BASIS_FUNCTIONS, NT_SOLVER, NX_SOLVER, L, T_FINAL, DT,
    INITIAL_STATE_VAL, SENSOR_LOCATIONS, X_GRID_SOLVER, ALPHA,
    SURROGATE_MODEL_PATH, CONTROLLER_MODEL_PATH, FIGURES_DIR
)
from pde_data import (
    solve_pde_time_varying, generate_grf_time_series, get_controller_dataloaders
)
from pde_surrogate import PropagatorDeepONet
from pde_controller import RecurrentController


def evaluate_surrogate():
    """
    Evaluate the surrogate model's multi-step rollout accuracy.

    Compares autoregressive predictions from the DeepONet with
    ground truth solutions from the Crank-Nicolson PDE solver.
    """
    print("\n" + "="*60)
    print("SURROGATE MODEL EVALUATION")
    print("="*60)

    # --- Load Surrogate ---
    if not os.path.exists(SURROGATE_MODEL_PATH):
        print(f"Surrogate not found at {SURROGATE_MODEL_PATH}. Please train first.")
        return

    surrogate = PropagatorDeepONet().to(DEVICE)
    surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    surrogate.eval()
    print("Loaded surrogate model.")

    # --- Generate Test Case ---
    print("\n--- Generating test case ---")
    np.random.seed(42)  # Reproducibility

    # Generate smooth control sequence
    length_scale = 1.5
    w_sequence = generate_grf_time_series(NT_SOLVER, NUM_BASIS_FUNCTIONS, length_scale)
    w_sequence = np.clip(w_sequence * 0.7, -1.0, 1.0)

    # Create basis functions and reconstruct u(x,t)
    basis_grid = np.cos(np.arange(NUM_BASIS_FUNCTIONS) * np.pi * X_GRID_SOLVER[:, None] / L)
    u_xt = w_sequence @ basis_grid.T  # (NT_SOLVER, NX_SOLVER)

    # Ground truth from PDE solver
    true_solution = solve_pde_time_varying(u_xt)  # (NT_SOLVER, NX_SOLVER)

    # --- Prepare for surrogate rollout ---
    sensor_x = SENSOR_LOCATIONS.numpy().flatten()
    basis_sensors = np.cos(np.arange(NUM_BASIS_FUNCTIONS) * np.pi * sensor_x[:, None] / L)
    basis_sensors_T = torch.from_numpy(basis_sensors.T).float().to(DEVICE)

    trunk_inputs = SENSOR_LOCATIONS.unsqueeze(0).to(DEVICE)
    w_tensor = torch.from_numpy(w_sequence).float().to(DEVICE)

    # --- Autoregressive Rollout ---
    print("--- Running autoregressive rollout ---")
    predicted_states = []
    current_state = torch.full((1, M_SENSORS), INITIAL_STATE_VAL, device=DEVICE)

    with torch.no_grad():
        for k in range(NT_SOLVER - 1):
            w_k = w_tensor[k:k+1, :]  # (1, NUM_BASIS)
            u_k = ALPHA * w_k @ basis_sensors_T  # (1, M_SENSORS)

            next_state = surrogate(current_state, u_k, trunk_inputs)
            predicted_states.append(next_state.squeeze().cpu().numpy())
            current_state = next_state

    predicted_solution = np.array(predicted_states)  # (NT_SOLVER-1, M_SENSORS)

    # --- Compute Metrics ---
    # Since sensors == grid, compare directly
    true_on_sensors = true_solution[1:, :]  # Skip t=0

    mse_per_step = np.mean((true_on_sensors - predicted_solution)**2, axis=1)
    final_mse = mse_per_step[-1]
    mean_mse = np.mean(mse_per_step)

    print(f"\nResults:")
    print(f"  Mean rollout MSE: {mean_mse:.4e}")
    print(f"  Final step MSE:   {final_mse:.4e}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Surrogate Model: Autoregressive Rollout Evaluation', fontsize=14)

    vmin, vmax = true_solution.min(), true_solution.max()

    # Ground truth
    im1 = axes[0].imshow(true_solution.T, extent=[0, T_FINAL, 0, L],
                         origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('Ground Truth (PDE Solver)')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Position x')
    plt.colorbar(im1, ax=axes[0])

    # Prediction (starts at t=DT)
    im2 = axes[1].imshow(predicted_solution.T, extent=[DT, T_FINAL, 0, L],
                         origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('DeepONet Prediction')
    axes[1].set_xlabel('Time t')
    plt.colorbar(im2, ax=axes[1])

    # Absolute error
    error = np.abs(true_on_sensors - predicted_solution)
    im3 = axes[2].imshow(error.T, extent=[DT, T_FINAL, 0, L],
                         origin='lower', aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Time t')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'surrogate_evaluation.png'), dpi=150)
    plt.show()


def evaluate_controller(target_type='ramp'):
    """
    Evaluate the trained controller's closed-loop performance.

    Args:
        target_type: One of 'constant', 'ramp', 'sine', 'gaussian'
    """
    print("\n" + "="*60)
    print("CONTROLLER EVALUATION")
    print("="*60)

    # --- Load Models ---
    if not os.path.exists(SURROGATE_MODEL_PATH):
        print(f"Surrogate not found. Please train first.")
        return
    if not os.path.exists(CONTROLLER_MODEL_PATH):
        print(f"Controller not found. Please train first.")
        return

    surrogate = PropagatorDeepONet().to(DEVICE)
    surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    surrogate.eval()

    controller = RecurrentController().to(DEVICE)
    controller.load_state_dict(torch.load(CONTROLLER_MODEL_PATH, map_location=DEVICE, weights_only=True))
    controller.eval()

    print("Loaded surrogate and controller models.")

    # --- Define Target Profile ---
    sensor_x = SENSOR_LOCATIONS.cpu().numpy().flatten()

    if target_type == 'constant':
        target_np = np.full(M_SENSORS, 1.0)
    elif target_type == 'ramp':
        target_np = np.linspace(0.5, 1.5, M_SENSORS)
    elif target_type == 'sine':
        target_np = 0.5 + 0.3 * np.sin(2 * np.pi * sensor_x / L)
    elif target_type == 'gaussian':
        center, width = L / 2, L / 8
        target_np = 1.0 + 0.5 * np.exp(-((sensor_x - center)**2) / (2 * width**2))
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    target_torch = torch.from_numpy(target_np).float().unsqueeze(0).to(DEVICE)
    print(f"\nTarget type: {target_type}")

    # --- Prepare for Simulation ---
    data_info = get_controller_dataloaders()
    basis_at_sensors_T = data_info['basis_at_sensors_T'].to(DEVICE)
    trunk_inputs = data_info['trunk_inputs'].to(DEVICE)

    # --- Closed-Loop Simulation ---
    print("--- Running closed-loop simulation ---")
    T_current = torch.full((1, M_SENSORS), INITIAL_STATE_VAL, device=DEVICE)
    hidden_state = None

    state_history = [T_current.cpu().numpy().flatten()]
    control_history = []

    with torch.no_grad():
        for k in range(NT_SOLVER - 1):
            # Controller decides action
            w_k, hidden_state = controller(T_current, target_torch, hidden_state)
            control_k = w_k @ basis_at_sensors_T

            control_history.append(w_k.cpu().numpy().flatten())

            # Surrogate predicts next state
            T_next = surrogate(T_current, control_k, trunk_inputs)
            T_current = T_next

            state_history.append(T_current.cpu().numpy().flatten())

    state_history = np.array(state_history)
    control_history = np.array(control_history)

    # --- Compute Metrics ---
    final_state = state_history[-1]
    final_mse = np.mean((final_state - target_np)**2)
    print(f"\nResults:")
    print(f"  Final state MSE: {final_mse:.4e}")

    # --- Visualization ---
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: State evolution heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(state_history.T, extent=[0, T_FINAL, 0, L],
                     origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title('State Evolution under Control')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position x')
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Final state vs target
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(sensor_x, state_history[0], 'b--', label='Initial (t=0)', linewidth=2)
    ax2.plot(sensor_x, target_np, 'k:', label='Target', linewidth=3)
    ax2.plot(sensor_x, final_state, 'r-', label='Final (t=T)', linewidth=2)
    ax2.set_title(f'Controller Performance (Target: {target_type})')
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: State trajectories at selected points
    ax3 = fig.add_subplot(2, 2, 3)
    time_axis = np.linspace(0, T_FINAL, NT_SOLVER)
    selected_indices = [0, M_SENSORS//4, M_SENSORS//2, 3*M_SENSORS//4, M_SENSORS-1]
    for idx in selected_indices:
        ax3.plot(time_axis, state_history[:, idx],
                label=f'x={sensor_x[idx]:.2f}', linewidth=1.5)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('State Trajectories at Selected Locations')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Control weights over time
    ax4 = fig.add_subplot(2, 2, 4)
    control_time = np.linspace(0, T_FINAL, NT_SOLVER - 1, endpoint=False)
    for i in range(NUM_BASIS_FUNCTIONS):
        ax4.plot(control_time, control_history[:, i], label=f'w_{i}')
    ax4.set_title('Control Weights over Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Weight Value')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'controller_evaluation_{target_type}.png'), dpi=150)
    plt.show()


def run_multiple_evaluations():
    """Run controller evaluation on multiple target types."""
    for target_type in ['constant', 'ramp', 'sine', 'gaussian']:
        evaluate_controller(target_type)
        print()


if __name__ == "__main__":
    # Evaluate surrogate
    evaluate_surrogate()

    # Evaluate controller on ramp target
    #evaluate_controller('ramp')

    # Uncomment to run all target types:
    run_multiple_evaluations()
