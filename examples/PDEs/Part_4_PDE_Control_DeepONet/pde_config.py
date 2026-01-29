# pde_config.py
# Central configuration file for the PDE Control with DeepONet example.
# This demonstrates learning neural PDE surrogates and control policies
# using the neuromancer library.

import os
import numpy as np
import torch

# --- Device ---
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# --- PDE & Simulation Parameters ---
# Heat equation with control: dT/dt = D * d^2T/dx^2 - BETA*(T - V_REF) + ALPHA*u(x,t)
L = 1.0                    # Domain length [0, L]
T_FINAL = 1.0              # Final simulation time
D = 0.01                   # Diffusion coefficient
BETA = 0.5                 # Reaction (leakage) coefficient
ALPHA = 2.0                # Control/source coefficient
INITIAL_STATE_VAL = 0.0    # T(x,0) = INITIAL_STATE_VAL
V_REF_VAL = 0.0            # Ambient/Reference value

# --- Discretization ---
NX_SOLVER = 41             # Number of spatial grid points
NT_SOLVER = 41             # Number of time steps
DT = T_FINAL / (NT_SOLVER - 1)

# --- Shared Grids ---
X_GRID_SOLVER = np.linspace(0.0, L, NX_SOLVER)  # (NX_SOLVER,)
# Sensors = solver grid (guarantees 1:1 alignment, no interpolation)
SENSOR_LOCATIONS = torch.from_numpy(X_GRID_SOLVER.copy()).view(-1, 1).float()  # (M_SENSORS, 1)
M_SENSORS = SENSOR_LOCATIONS.shape[0]

# --- Control Representation ---
# Control is represented as: u(x,t) = sum_i w_i(t) * cos(i*pi*x/L)
NUM_BASIS_FUNCTIONS = 6

# --- Data Generation ---
NUM_TRAIN_SIMULATIONS = 800
NUM_TEST_SIMULATIONS = 200
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_trajectories.npz")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_trajectories.npz")

# --- Model Architecture ---
# DeepONet branch network
BRANCH_HIDDEN_DIMS = [256, 256]
DEEPONET_OUTPUT_DIM = 128

# DeepONet trunk network
TRUNK_INPUT_DIM = 1  # x coordinate
TRUNK_HIDDEN_DIM = 128

# Controller LSTM
CONTROLLER_HIDDEN_DIM = 256
CONTROLLER_NUM_LAYERS = 2

# --- Surrogate Training ---
SURROGATE_LEARNING_RATE = 1e-3
SURROGATE_EPOCHS = 500
SURROGATE_BATCH_SIZE = 128
SURROGATE_PATIENCE = 50

# --- Controller Training ---
CONTROLLER_LEARNING_RATE = 1e-3
CONTROLLER_EPOCHS = 1000
CONTROLLER_BATCH_SIZE = 256

# Loss weights for controller training
TERMINAL_LOSS_WEIGHT = 1.0      # Weight on final state matching target
TRACKING_LOSS_WEIGHT = 0.1      # Weight on intermediate state tracking
EFFORT_LOSS_WEIGHT = 1e-5       # Weight on control effort penalty

# --- Paths ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
SURROGATE_MODEL_PATH = os.path.join(MODELS_DIR, "propagator_deeponet.pth")
CONTROLLER_MODEL_PATH = os.path.join(MODELS_DIR, "recurrent_controller.pth")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
