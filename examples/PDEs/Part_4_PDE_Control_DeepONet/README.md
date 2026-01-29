# Part 4: PDE Control with DeepONet

This example demonstrates learning neural network-based PDE surrogates and control policies using NeuroMANCER. We solve a heat equation control problem using a two-phase approach:

1. **Surrogate Learning**: Train a DeepONet to approximate one-step PDE dynamics
2. **Controller Learning**: Train an LSTM-based controller using differentiable rollouts through the frozen surrogate

## Problem Formulation

We consider a 1D heat equation with reaction and distributed control:

$$\frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial x^2} - \beta (T - T_{ref}) + \alpha \cdot u(x,t)$$

with:
- Domain: $x \in [0, L]$, $t \in [0, T_{final}]$
- Neumann boundary conditions (zero flux)
- Initial condition: $T(x, 0) = 0$

The control input $u(x,t)$ is parameterized using cosine basis functions:

$$u(x,t) = \sum_{i=0}^{N_{basis}-1} w_i(t) \cos\left(\frac{i \pi x}{L}\right)$$

**Control Objective**: Drive the temperature field from the initial state to a target profile $T_{target}(x)$ at the final time.

## Architecture

### DeepONet Surrogate (Propagator)

The surrogate learns to predict one-step dynamics: $(T_k, u_k) \rightarrow T_{k+1}$

```
Branch Network: [T_k || u_k] → 256 → 256 → 128
Trunk Network:  x → 128 → 128 → 128
Output: einsum(branch, trunk) + bias
```

### LSTM Controller

The controller generates control weights given the current and target states:

```
Input: [T_k || T_target]
LSTM: 2 layers, hidden_dim=256
Output Head: 256 → 128 → NUM_BASIS_FUNCTIONS (with Tanh)
```

## Two-Phase Training

### Phase 1: Surrogate Training
- Generate trajectories by solving the PDE with random control sequences
- Train the DeepONet to minimize one-step prediction error
- Use NeuroMANCER's `Node`, `variable`, `PenaltyLoss`, `Problem`, and `Trainer`

### Phase 2: Controller Training
- Freeze the trained surrogate
- Self-supervised training: generate random targets, rollout closed-loop system
- Backpropagate through the entire trajectory to learn control policy
- Multi-objective loss: terminal error + tracking error + control effort

## Files

| File | Description |
|------|-------------|
| `pde_config.py` | Central configuration (PDE parameters, architecture, hyperparameters) |
| `pde_data.py` | PDE solver, data generation, and DataLoader utilities |
| `pde_surrogate.py` | DeepONet model definition and training |
| `pde_controller.py` | LSTM controller definition and training |
| `pde_evaluate.py` | Evaluation and visualization utilities |

## Usage

Run the scripts in order:

```bash
# 1. Generate training data
python pde_data.py

# 2. Train the DeepONet surrogate
python pde_surrogate.py

# 3. Train the LSTM controller
python pde_controller.py

# 4. Evaluate both models
python pde_evaluate.py
```

## Expected Results

### Surrogate
- One-step prediction MSE should be < 1e-4
- Multi-step rollout should closely track ground truth PDE solution

### Controller
- Terminal loss should decrease significantly during training
- Controller should successfully drive the system to various target profiles
- Control weights should be smooth and bounded

## NeuroMANCER Features Used

- `Node`: Wrap PyTorch modules into symbolic computational nodes
- `variable`: Create symbolic variables for loss computation
- `PenaltyLoss`: Combine multiple objectives and constraints
- `Problem`: Define the optimization problem
- `Trainer`: Handle training loops, early stopping, logging
- `DictDataset`: Create datasets from dictionaries

## Configuration

Key parameters in `pde_config.py`:

```python
# PDE Parameters
L = 1.0              # Domain length
T_FINAL = 1.0        # Simulation time
D = 0.01             # Diffusion coefficient
BETA = 0.5           # Reaction coefficient
ALPHA = 2.0          # Control coefficient

# Discretization
NX_SOLVER = 41       # Spatial grid points
NT_SOLVER = 41       # Time steps

# Training
SURROGATE_EPOCHS = 500
CONTROLLER_EPOCHS = 1000
```

## References

- [DeepONet: Learning nonlinear operators](https://www.nature.com/articles/s42256-021-00302-5)
- [Differentiable Predictive Control](https://www.sciencedirect.com/science/article/pii/S0959152422000981)
- [Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
