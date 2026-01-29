"""
pde_controller.py
Train the LSTM-based recurrent controller using differentiable rollouts.

This script demonstrates using neuromancer for self-supervised control learning:
1. Load a pre-trained DeepONet surrogate (frozen)
2. Train an LSTM controller to drive the system to random target states
3. Use differentiable rollouts through the frozen surrogate for end-to-end training

The controller learns by:
- Generating random target temperature profiles
- Rolling out the closed-loop system (controller + surrogate)
- Backpropagating through the entire trajectory to minimize tracking error
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from neuromancer.system import Node
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

from pde_config import (
    DEVICE, M_SENSORS, NUM_BASIS_FUNCTIONS, NT_SOLVER, L,
    CONTROLLER_HIDDEN_DIM, CONTROLLER_NUM_LAYERS,
    CONTROLLER_LEARNING_RATE, CONTROLLER_EPOCHS, CONTROLLER_BATCH_SIZE,
    TERMINAL_LOSS_WEIGHT, TRACKING_LOSS_WEIGHT, EFFORT_LOSS_WEIGHT,
    INITIAL_STATE_VAL, SURROGATE_MODEL_PATH, CONTROLLER_MODEL_PATH,
    SENSOR_LOCATIONS, FIGURES_DIR
)
from pde_data import get_controller_dataloaders
from pde_surrogate import PropagatorDeepONet


class RecurrentController(nn.Module):
    """
    An LSTM-based controller for time-varying PDE control.

    At each step k, the controller takes:
    - T_k: current state at sensor locations
    - T_target: desired target state
    - hidden: LSTM hidden state from previous step

    And outputs:
    - w_k: control basis function weights in [-1, 1]
    - new_hidden: updated LSTM hidden state
    """

    def __init__(self):
        super(RecurrentController, self).__init__()

        # Input: concatenated [current_state, target_state]
        controller_input_dim = M_SENSORS + M_SENSORS

        # LSTM processes the sequence and maintains memory
        self.lstm = nn.LSTM(
            input_size=controller_input_dim,
            hidden_size=CONTROLLER_HIDDEN_DIM,
            num_layers=CONTROLLER_NUM_LAYERS,
            batch_first=True
        )

        # Output head maps LSTM output to control weights
        self.output_head = nn.Sequential(
            nn.Linear(CONTROLLER_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_BASIS_FUNCTIONS),
            nn.Tanh()  # Constrains output to [-1, 1]
        )

    def forward(self, T_k, T_target, hidden_state=None):
        """
        One step of control decision.

        Args:
            T_k: Current state, shape (batch_size, M_SENSORS)
            T_target: Target state, shape (batch_size, M_SENSORS)
            hidden_state: Previous LSTM hidden state (h, c) or None

        Returns:
            w_k: Control weights, shape (batch_size, NUM_BASIS_FUNCTIONS)
            new_hidden_state: Updated LSTM hidden state
        """
        # Concatenate current and target states
        lstm_input = torch.cat([T_k, T_target], dim=1)  # (B, 2*M_SENSORS)

        # LSTM expects (batch, seq_len, features), seq_len=1
        lstm_input = lstm_input.unsqueeze(1)

        # LSTM forward pass
        lstm_out, new_hidden_state = self.lstm(lstm_input, hidden_state)

        # Remove sequence dimension
        lstm_out_squeezed = lstm_out.squeeze(1)  # (B, hidden_dim)

        # Output head produces control weights
        w_k = self.output_head(lstm_out_squeezed)  # (B, NUM_BASIS_FUNCTIONS)

        return w_k, new_hidden_state


def generate_random_targets(batch_size, device):
    """
    Generate a batch of random, smooth target temperature profiles.

    The targets are combinations of sinusoidal waves centered around 0.5.

    Args:
        batch_size: Number of targets to generate
        device: torch device

    Returns:
        T_target: shape (batch_size, M_SENSORS)
    """
    sensor_x = SENSOR_LOCATIONS.squeeze().to(device)  # (M_SENSORS,)
    targets = torch.zeros(batch_size, M_SENSORS, device=device)

    for i in range(batch_size):
        num_waves = np.random.randint(1, 4)
        for _ in range(num_waves):
            amplitude = torch.randn(1, device=device).item() * 0.5
            frequency = torch.randn(1, device=device).item() * 3.0
            phase = torch.rand(1, device=device).item() * 2 * np.pi
            targets[i, :] += amplitude * torch.sin(frequency * sensor_x * np.pi + phase)

    # Center around 0.5, scale to avoid extremes
    targets = 0.5 + targets * 0.4
    return targets


def train_controller():
    """Main function to train the recurrent controller."""
    print(f"Training Controller on device: {DEVICE}")

    # --- Load Frozen Surrogate ---
    if not os.path.exists(SURROGATE_MODEL_PATH):
        raise FileNotFoundError(
            f"Surrogate model not found at {SURROGATE_MODEL_PATH}. "
            "Please run pde_surrogate.py first."
        )

    surrogate = PropagatorDeepONet().to(DEVICE)
    surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    surrogate.eval()
    for param in surrogate.parameters():
        param.requires_grad = False
    print("Loaded and froze pre-trained surrogate model.")

    # --- Initialize Controller ---
    controller = RecurrentController().to(DEVICE)
    optimizer = torch.optim.Adam(controller.parameters(), lr=CONTROLLER_LEARNING_RATE)

    # --- Get basis functions and trunk inputs ---
    data_info = get_controller_dataloaders()
    basis_at_sensors_T = data_info['basis_at_sensors_T'].to(DEVICE)  # (NUM_BASIS, M_SENSORS)
    trunk_inputs = data_info['trunk_inputs'].to(DEVICE)  # (1, M_SENSORS, 1)

    # --- Training Loop ---
    print("\n--- Starting Controller Training ---")
    print(f"Training for {CONTROLLER_EPOCHS} epochs with batch size {CONTROLLER_BATCH_SIZE}")

    loss_history = {
        'total': [], 'terminal': [], 'tracking': [], 'effort': []
    }

    for epoch in range(CONTROLLER_EPOCHS):
        controller.train()

        # Generate random targets for this epoch
        T_target = generate_random_targets(CONTROLLER_BATCH_SIZE, DEVICE)

        # Initialize simulation
        T_current = torch.full(
            (CONTROLLER_BATCH_SIZE, M_SENSORS),
            INITIAL_STATE_VAL,
            device=DEVICE
        )
        hidden_state = None

        # Accumulators for loss terms
        total_effort = 0.0
        tracking_loss = 0.0

        # --- Differentiable Rollout ---
        for k in range(NT_SOLVER - 1):
            # Controller decides control action
            w_k, hidden_state = controller(T_current, T_target, hidden_state)

            # Reconstruct control at sensor locations
            control_k = w_k @ basis_at_sensors_T  # (B, M_SENSORS)

            # Accumulate effort cost
            total_effort += torch.mean(torch.sum(control_k**2, dim=1))

            # Surrogate predicts next state
            trunk_expanded = trunk_inputs.expand(CONTROLLER_BATCH_SIZE, -1, -1)
            T_next = surrogate(T_current, control_k, trunk_expanded)

            # Update state
            T_current = T_next

            # Accumulate tracking loss
            tracking_loss += torch.mean((T_current - T_target)**2)

        # --- Compute Final Loss ---
        T_final = T_current

        # Terminal loss: how close is final state to target
        terminal_loss = torch.mean((T_final - T_target)**2)

        # Normalize by number of steps
        avg_tracking_loss = tracking_loss / (NT_SOLVER - 1)
        avg_effort = total_effort / (NT_SOLVER - 1)

        # Weighted total loss
        total_loss = (
            TERMINAL_LOSS_WEIGHT * terminal_loss +
            TRACKING_LOSS_WEIGHT * avg_tracking_loss +
            EFFORT_LOSS_WEIGHT * avg_effort
        )

        # --- Backpropagation ---
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()

        # --- Record losses ---
        loss_history['total'].append(total_loss.item())
        loss_history['terminal'].append(terminal_loss.item())
        loss_history['tracking'].append(avg_tracking_loss.item())
        loss_history['effort'].append(avg_effort.item())

        # --- Logging ---
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{CONTROLLER_EPOCHS} | "
                  f"Total: {total_loss.item():.4f} | "
                  f"Terminal: {terminal_loss.item():.4f} | "
                  f"Tracking: {avg_tracking_loss.item():.4f} | "
                  f"Effort: {avg_effort.item():.2f}")

    # --- Save Controller ---
    os.makedirs(os.path.dirname(CONTROLLER_MODEL_PATH), exist_ok=True)
    torch.save(controller.state_dict(), CONTROLLER_MODEL_PATH)
    print(f"\nController saved to {CONTROLLER_MODEL_PATH}")

    # --- Plot Loss Curves ---
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['total'], label='Total Loss', linewidth=2)
    plt.plot(loss_history['terminal'], label='Terminal Loss', alpha=0.7)
    plt.plot(loss_history['tracking'], label='Tracking Loss', alpha=0.7)
    plt.plot(np.array(loss_history['effort']) * EFFORT_LOSS_WEIGHT,
             label='Effort Loss (weighted)', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Controller Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'controller_training_loss.png'), dpi=150)
    plt.show()
    print(f"Loss curves saved to {FIGURES_DIR}/controller_training_loss.png")


if __name__ == "__main__":
    train_controller()
