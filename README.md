# Connect Four AI

A terminal-based implementation of Connect Four in Python featuring two distinct AI opponents: **Monte Carlo Simulations** and an **Actor-Critic Residual CNN** trained via Deep Reinforcement Learning.

---

## Quickstart

### Installation & Run

```bash
# Clone the repository
git clone https://github.com/yezdata/connect_four_ai.git
cd connect_four_ai

# Set up virtual environment & dependencies
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Play against Actor-Critic RL CNN (default 8x8 board, human moves first)
python main.py --ai cnn --human-first

# Play against Monte Carlo AI (6x6 board)
python main.py --ai mc --rows 6 --cols 6
```

### Command-Line Arguments (`main.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ai` | `str` | `mc` | AI strategy: `mc` (Monte Carlo) or `cnn` (Actor-Critic RL CNN). |
| `--rows` | `int` | `8` | Number of board rows. *(Must be `8` when using `--ai cnn`)* |
| `--cols` | `int` | `8` | Number of board columns. *(Must be `8` when using `--ai cnn`)* |
| `--human-first` | `flag` | `False` | Pass this flag to let the human play first (`X`). Default is AI first (`X`). |


## Game Rules
- Standard Connect Four rules with gravity (pieces drop to the lowest empty row in the chosen column).
- First player to connect 4 consecutive symbols horizontally, vertically, or diagonally wins.
---

## AI Strategies

### 1. Monte Carlo Simulation (`--ai mc`)
Evaluates candidate legal moves using statistical rollouts:
- Simulates random games (`sim_count=200`) to completion for each valid column.
- Selects the column yielding the highest empirical win rate.
- Board-agnostic (supports arbitrary board dimensions).

### 2. Actor-Critic RL CNN (`--ai cnn`)
An AlphaZero-inspired dual-headed convolutional network that evaluates spatial board topology and outputs both action distributions and state evaluations in a single forward pass.

#### Model Architecture
- **Perspective-Invariant Input Encoding:** The board is encoded as a 2-channel binary tensor $(2 \times H \times W)$ relative to the active player (Channel 0: own pieces, Channel 1: opponent pieces). This canonical representation makes policy and value evaluations invariant to player symbol or turn order.
- **Deep Residual Backbone:** A stack of 10 residual blocks (128 channels, $3\times3$ convolutions) with Group Normalization and identity skip connections. The receptive field spans the entire grid, allowing the network to capture complex, non-local spatial patterns—such as multi-turn forks, diagonal alignments, and open-ended threats—without vanishing gradients.
- **Decoupled Dual Heads:**
  - **Policy Head (Actor):** Compresses shared feature maps through a $1\times1$ convolutional bottleneck down to column action logits. Illegal moves (full columns) are masked to $-\infty$ prior to softmax, guaranteeing valid action selection. At runtime, actions are drawn via temperature scaling / top-$k$ sampling, or chosen deterministically via argmax.
  - **Value Head (Critic):** Routes spatial features through a $1\times1$ bottleneck into an MLP with Layer Normalization and $\tanh$ activation, predicting the expected terminal game outcome $V(s) \in [-1, 1]$ directly from the current player's perspective.

#### Training Pipeline (`rl_cnn/train_batched_actor.py`)
Trained entirely through self-play reinforcement learning without human demonstrations or heuristic guidance:
- **Vectorized Parallel Rollouts:** `BatchedConnectFourEnv` executes **1,024 concurrent game environments** directly on device (GPU/MPS/CPU), maximizing hardware throughput. Both players share the same network weights.
- **Advantage Actor-Critic (A2C) Objective:** Trajectories from completed games are collected retrospectively ($+1.0$ win, $-1.0$ loss, $0.0$ tie). The Critic's state estimates serve as a variance-reducing baseline:
  $$\mathcal{L} = -\mathbb{E} [\log \pi(a \mid s) \cdot (R - V(s))] + c_{\text{val}} \cdot \text{MSE}(V(s), R) - c_{\text{ent}} \cdot \mathcal{H}(\pi(\cdot \mid s))$$
  Advantage scaling $(R - V(s))$ stabilizes policy gradient updates, while entropy regularization prevents premature convergence on rigid opening patterns.
- **Gradient Accumulation & Optimization:** Completed trajectories are pooled, shuffled, and processed across minibatches using gradient accumulation to maintain a stable effective batch size despite variable game lengths.

#### Training Hyperparameters
The default configuration in `rl_cnn/train_batched_actor.py`:

| Parameter | Value | Description |
|---|---|---|
| `num_envs` | `1024` | Parallel simulation environments on device |
| `batch_update_games` | `128` | Completed games buffered per optimization update |
| `minibatch_size` | `2048` | Minibatch chunk size for gradient accumulation |
| `total_completed_games` | `10,000,000` | Target training budget |
| `lr` | `1e-4` | Base learning rate (`AdamW`, weight decay `1e-4`) |
| `lr_scheduler` | Cosine Annealing | Anneals learning rate down to `1e-6` |
| `gamma` ($\gamma$) | `1.0` | Reward discount factor (no discounting) |
| `value_coef` ($c_{\text{val}}$) | `0.5` | Value loss weighting |
| `entropy_coef` ($c_{\text{ent}}$) | `0.05` | Policy entropy bonus coefficient |
| Board Dimensions | `8 x 8` | Grid size trained by the default checkpoint |

---

## Project Structure

```
connect_four_ai/
├── main.py                  # CLI entry point and game loop
├── game_engine.py           # Connect Four rules, board state, ANSI rendering
├── requirements.txt
├── ai_strategy/
│   ├── monte_carlo.py       # Monte Carlo rollout strategy
│   └── rl_cnn.py            # RL model wrapper, inference & action sampling
└── rl_cnn/
    ├── model.py             # PyTorch models (ActorCriticCNN, ResBlock, old model architecture:PolicyCNN)
    ├── train_batched_actor.py # Batched self-play Actor-Critic training pipeline
    └── train_... # Old PolicyCNN train / Old single concurrent env traing
    └── models/              # Pretrained weights (connect4_actor_critic_r8_c8.pt)
```

---

