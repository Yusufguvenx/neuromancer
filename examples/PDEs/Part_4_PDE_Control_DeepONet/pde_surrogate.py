"""
pde_surrogate.py
Train the DeepONet surrogate model for one-step PDE dynamics prediction.

This script demonstrates using neuromancer to train a DeepONet that learns
to predict T_{k+1} = f(T_k, u_k, x) where:
- T_k is the current temperature field at sensor locations
- u_k is the control input at sensor locations
- x are the spatial locations where we want to predict

The architecture is a branch-trunk DeepONet:
- Branch: processes [T_k || u_k] to produce latent features
- Trunk: processes spatial locations x
- Output: inner product of branch and trunk outputs
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from neuromancer.system import Node
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

from pde_config import (
    DEVICE, M_SENSORS, DEEPONET_OUTPUT_DIM, BRANCH_HIDDEN_DIMS,
    TRUNK_INPUT_DIM, TRUNK_HIDDEN_DIM, SURROGATE_LEARNING_RATE,
    SURROGATE_EPOCHS, SURROGATE_PATIENCE, SURROGATE_MODEL_PATH,
    FIGURES_DIR
)
from pde_data import get_surrogate_dataloaders


class PropagatorDeepONet(nn.Module):
    """
    A DeepONet that learns a one-step time advancement operator for PDEs.

    Architecture:
        Branch Net: [T_k (M_SENSORS) || u_k (M_SENSORS)] -> hidden -> output_dim
        Trunk Net: x_locations (1) -> hidden -> output_dim
        Output: einsum('bi,bsi->bs', branch_out, trunk_out) + bias

    This predicts T_{k+1}(x) given T_k and u_k.
    """

    def __init__(self):
        super(PropagatorDeepONet, self).__init__()

        # Branch input: concatenated [T_k, u_k]
        branch_input_size = M_SENSORS + M_SENSORS

        # Build branch network
        branch_layers = []
        in_dim = branch_input_size
        for hidden_dim in BRANCH_HIDDEN_DIMS:
            branch_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        branch_layers.append(nn.Linear(in_dim, DEEPONET_OUTPUT_DIM))

        self.branch = nn.Sequential(*branch_layers)

        # Build trunk network
        self.trunk = nn.Sequential(
            nn.Linear(TRUNK_INPUT_DIM, TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, DEEPONET_OUTPUT_DIM)
        )

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, T_k, u_k, x_locs):
        """
        Forward pass of the DeepONet.

        Args:
            T_k: Current state at sensors, shape (batch_size, M_SENSORS)
            u_k: Control input at sensors, shape (batch_size, M_SENSORS)
            x_locs: Spatial locations, shape (batch_size, M_SENSORS, 1)

        Returns:
            T_k_plus_1: Predicted next state, shape (batch_size, M_SENSORS)
        """
        # Concatenate state and control for branch input
        branch_input = torch.cat([T_k, u_k], dim=1)  # (B, 2*M_SENSORS)
        branch_out = self.branch(branch_input)  # (B, output_dim)

        # Trunk processes spatial locations
        trunk_out = self.trunk(x_locs)  # (B, M_SENSORS, output_dim)

        # DeepONet output: inner product + bias
        output = torch.einsum('bi,bsi->bs', branch_out, trunk_out)  # (B, M_SENSORS)

        return output + self.bias


def create_surrogate_node():
    """
    Create a neuromancer Node wrapping the PropagatorDeepONet.

    Returns:
        Node with input keys ['T_k', 'u_k', 'x_locs'] and output key ['T_k_plus_1']
    """
    model = PropagatorDeepONet()
    node = Node(model, ['T_k', 'u_k', 'x_locs'], ['T_k_plus_1'], name='DeepONet')
    return node


def move_batch_to_device(batch, device):
    """Move batch tensors to device, keeping non-tensor values unchanged."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_surrogate(problem, train_loader, test_loader, optimizer, epochs, patience, device):
    """
    Custom training loop for the surrogate model.

    Uses neuromancer Problem but with explicit training loop for loss tracking.

    Returns:
        best_model: state dict of best model
        train_losses: list of training losses per epoch
        test_losses: list of test losses per epoch
    """
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model = deepcopy(problem.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        problem.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            output = problem(batch)

            optimizer.zero_grad()
            output['train_loss'].backward()
            torch.nn.utils.clip_grad_norm_(problem.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += output['train_loss'].item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation
        problem.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = move_batch_to_device(batch, device)
                output = problem(batch)
                # Test data has 'test_loss' key due to DictDataset naming
                loss_key = 'test_loss' if 'test_loss' in output else 'train_loss'
                epoch_test_loss += output[loss_key].item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Early stopping check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model = deepcopy(problem.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4e} | Test: {avg_test_loss:.4e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_model, train_losses, test_losses


def main():
    """Main training function for the DeepONet surrogate."""
    print(f"Training DeepONet Surrogate on device: {DEVICE}")

    # --- Data Setup ---
    train_loader, test_loader = get_surrogate_dataloaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # --- Model Setup ---
    surrogate_node = create_surrogate_node()
    print(f"Surrogate inputs: {surrogate_node.input_keys}")
    print(f"Surrogate outputs: {surrogate_node.output_keys}")

    # --- Define Loss using neuromancer variables ---
    # Predicted next state from model
    T_pred = variable('T_k_plus_1')
    # Ground truth next state from data
    T_true = variable('T_target')

    # MSE loss: (T_pred - T_true)^2
    surrogate_loss = (T_pred == T_true) ^ 2
    surrogate_loss.update_name('surrogate_mse')

    # Create PenaltyLoss (objectives list, constraints list)
    loss = PenaltyLoss([surrogate_loss], [])

    # --- Create Problem ---
    problem = Problem(
        nodes=[surrogate_node],
        loss=loss
    ).to(DEVICE)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(problem.parameters(), lr=SURROGATE_LEARNING_RATE)

    # --- Train ---
    print("\n--- Starting Surrogate Training ---")
    best_model, train_losses, test_losses = train_surrogate(
        problem, train_loader, test_loader, optimizer,
        SURROGATE_EPOCHS, SURROGATE_PATIENCE, DEVICE
    )

    # Load best model
    problem.load_state_dict(best_model)

    # --- Save Model ---
    os.makedirs(os.path.dirname(SURROGATE_MODEL_PATH), exist_ok=True)

    # Save just the DeepONet state dict for easy loading
    deeponet = problem.nodes[0].callable
    torch.save(deeponet.state_dict(), SURROGATE_MODEL_PATH)
    print(f"\nModel saved to {SURROGATE_MODEL_PATH}")

    # --- Final Evaluation ---
    print("\n--- Final Test Set Evaluation ---")
    print(f"Best test MSE: {min(test_losses):.4e}")

    # --- Plot Training Curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Surrogate Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'surrogate_training_loss.png'), dpi=150)
    plt.show()
    print(f"Training curve saved to {FIGURES_DIR}/surrogate_training_loss.png")


if __name__ == "__main__":
    main()
