# TODO: vizualizovat (show_state) posledni state vzdy po printu + i v train_single_env

import os
from typing import List, Tuple
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

import game_engine as ge
from rl_cnn.model import PolicyCNN


class BatchedConnectFourEnv:
    """Manages N concurrent Connect Four game environments with mixed opponent types."""

    def __init__(self, num_envs: int = 256, rows: int = 8, cols: int = 8):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.symbols = ["X", "O"]

        # Opponent configuration:
        # 0: Pure Self-Play (Model vs Model)
        # 1: Model is X (turn 0), Random Agent is O (turn 1)
        # 2: Random Agent is X (turn 0), Model is O (turn 1)
        self.env_types = [0] * num_envs
        quarter = num_envs // 4
        half = num_envs // 2

        for i in range(half, half + quarter):
            self.env_types[i] = 1
        for i in range(half + quarter, num_envs):
            self.env_types[i] = 2

        self.reset_all()

    def reset_all(self):
        self.boards = [
            [[" " for _ in range(self.cols)] for _ in range(self.rows)]
            for _ in range(self.num_envs)
        ]
        self.turn_idx = [0 for _ in range(self.num_envs)]
        # History stores only the Neural Network's snapshots: (state, mask, action, turn_idx)
        self.histories: List[List[Tuple[torch.Tensor, torch.Tensor, int, int]]] = [
            [] for _ in range(self.num_envs)
        ]

    def reset_single(self, env_idx: int):
        self.boards[env_idx] = [
            [" " for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.turn_idx[env_idx] = 0
        self.histories[env_idx] = []

    def is_random_turn(self, env_idx: int) -> bool:
        """Determines if the current turn in the given env belongs to the random agent."""
        etype = self.env_types[env_idx]
        cur_turn = self.turn_idx[env_idx]
        if etype == 1 and cur_turn == 1:
            return True
        if etype == 2 and cur_turn == 0:
            return True
        return False

    def get_batched_states_and_masks(
        self, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates board tensors directly on CPU."""
        states_tensor = torch.zeros(
            (self.num_envs, 2, self.rows, self.cols), dtype=torch.float32, device=device
        )
        masks_tensor = torch.zeros(
            (self.num_envs, self.cols), dtype=torch.bool, device=device
        )

        for i in range(self.num_envs):
            board = self.boards[i]
            cur_sym = self.symbols[self.turn_idx[i]]
            opp_sym = self.symbols[1 - self.turn_idx[i]]

            # Channel 0: current player tokens, Channel 1: opponent tokens
            for r in range(self.rows):
                for c in range(self.cols):
                    cell = board[r][c]
                    if cell == cur_sym:
                        states_tensor[i, 0, r, c] = 1.0
                    elif cell == opp_sym:
                        states_tensor[i, 1, r, c] = 1.0

            for c in range(self.cols):
                if ge.valid_move(c, board):
                    masks_tensor[i, c] = True

        return states_tensor, masks_tensor


def train_batched(
    total_completed_games: int = 10000,
    num_envs: int = 64,
    rows: int = 8,
    cols: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma: float = 0.95,
    entropy_coef: float = 0.01,
    batch_update_games: int = 64,
    save_path: str = "rl_cnn/connect4_policy.pt",
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Training on device: {device} | Parallel environments: {num_envs}")

    model = PolicyCNN(rows=rows, cols=cols).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    env = BatchedConnectFourEnv(num_envs=num_envs, rows=rows, cols=cols)

    completed_games_count = 0
    stats = {"X": 0, "O": 0, "Tie": 0}
    finished_trajectories = []

    updates_count = 0

    while completed_games_count < total_completed_games:
        # 1. Rollout phase: pure inference
        model.eval()
        with torch.no_grad():
            states, masks = env.get_batched_states_and_masks(device=device)
            logits = model(
                states, valid_mask=masks
            )  # parallel forward pass across all envs
            dist = Categorical(logits=logits)
            actions = dist.sample()  # Shape: (num_envs,)

        # 2. Step the environments
        for i in range(num_envs):
            cur_turn = env.turn_idx[i]
            board = env.boards[i]
            is_random = env.is_random_turn(i)

            if is_random:
                # Random agent: pick uniformly from valid columns
                valid_cols = torch.where(masks[i])[0].tolist()
                act = random.choice(valid_cols)
            else:
                act = actions[i].item()
                # Store snapshot ONLY when the neural network takes an action
                env.histories[i].append((states[i], masks[i], act, cur_turn))

            ge.make_move(act, board, env.symbols[cur_turn])

            # Check if this particular environment reached a terminal state
            if ge.is_won(board):
                winner = cur_turn
                stats["X" if winner == 0 else "O"] += 1
                if env.histories[i]:  # Only train if the model made at least one move
                    finished_trajectories.append((env.histories[i], winner))
                env.reset_single(i)
                completed_games_count += 1
            elif ge.is_tie(board):
                stats["Tie"] += 1
                if env.histories[i]:
                    finished_trajectories.append((env.histories[i], -1))
                env.reset_single(i)
                completed_games_count += 1
            else:
                # Switch active turn for this game
                env.turn_idx[i] = 1 - env.turn_idx[i]

        # 3. Training phase: reconstruct graph and update parameters
        if len(finished_trajectories) >= batch_update_games:
            b_states = []
            b_masks = []
            b_actions = []
            b_returns = []

            for history, winner in finished_trajectories:
                n_steps = len(history)
                for step, (s, m, a, p_idx) in enumerate(history):
                    if winner == -1:
                        ret = 0.0
                    else:
                        steps_to_end = n_steps - step - 1
                        base = 1.0 if p_idx == winner else -1.0
                        ret = base * (gamma**steps_to_end)

                    b_states.append(s)
                    b_masks.append(m)
                    b_actions.append(a)
                    b_returns.append(ret)

            # Stack into large training tensors
            b_states = torch.stack(b_states).to(device)
            b_masks = torch.stack(b_masks).to(device)
            b_actions = torch.tensor(b_actions, dtype=torch.long, device=device)
            b_returns = torch.tensor(b_returns, dtype=torch.float32, device=device)

            # Build autograd graph on the full batch at once
            model.train()
            logits = model(b_states, valid_mask=b_masks)
            dist = Categorical(logits=logits)

            action_log_probs = dist.log_prob(b_actions)

            probs = dist.probs
            log_probs_all = torch.log(probs + 1e-12)
            entropies = -(probs * log_probs_all).sum(dim=-1)

            policy_loss = -(action_log_probs * b_returns).mean()
            entropy_loss = -entropy_coef * entropies.mean()
            loss = policy_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            finished_trajectories.clear()

            updates_count += 1
            if updates_count % 10 == 0:
                print(
                    f"Games: {completed_games_count:6d}/{total_completed_games} | "
                    f"Wins -> X: {stats['X']}, O: {stats['O']}, Ties: {stats['Tie']} | "
                    f"Loss: {loss.item():.4f} (Pol: {policy_loss.item():.4f}, Ent: {entropies.mean().item():.4f})"
                )
                stats = {"X": 0, "O": 0, "Tie": 0}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining completed. Weights successfully saved to: {save_path}")


if __name__ == "__main__":
    train_batched(
        total_completed_games=50_000,
        num_envs=256,
        rows=8,
        cols=8,
        lr=8e-4,
        entropy_coef=0.03,
        gamma=1.0,
        batch_update_games=256,
    )
