import os
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical
from typing import List

import game_engine as ge
from rl_cnn.model import PolicyCNN, state_to_tensor


def play_self_play_game(
    model: PolicyCNN, rows: int = 8, cols: int = 8, device: str = "cpu"
):
    """
    Play 1 game, where the CNN play for both players.
    Saves log_probs of the moves and entropy.
    """
    state = [[" " for _ in range(cols)] for _ in range(rows)]
    symbols = ["X", "O"]
    turn_idx = 0  # 0 for X, 1 for O

    # save: (turn_idx_who_played, log_prob, entropy)
    history = []

    while not ge.is_tie(state):
        current_sym = symbols[turn_idx]
        opponent_sym = symbols[1 - turn_idx]

        # from current player pov
        s_tensor = state_to_tensor(state, current_sym, opponent_sym, device=device)

        # mask of valid moves
        valid_moves = [ge.valid_move(c, state) for c in range(cols)]
        mask_tensor = torch.tensor([valid_moves], dtype=torch.bool, device=device)

        logits = model(s_tensor, valid_mask=mask_tensor)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        chosen_col = action.item()
        ge.make_move(chosen_col, state, current_sym)
        history.append((turn_idx, log_prob, entropy))

        if ge.is_won(state):
            winner_idx = turn_idx
            return history, winner_idx

        turn_idx = 1 - turn_idx

    # tie
    return history, -1


def train(
    num_episodes: int = 5000,
    rows: int = 8,
    cols: int = 8,
    lr: float = 1e-3,
    gamma: float = 0.95,
    entropy_coef: float = 0.01,
    save_path: str = "rl_cnn/connect4_policy.pt",
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = PolicyCNN(rows=rows, cols=cols).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    win_stats = {"X": 0, "O": 0, "Tie": 0}

    for episode in range(1, num_episodes + 1):
        model.train()
        history, winner_idx = play_self_play_game(model, rows, cols, device=device)

        if winner_idx == 0:
            win_stats["X"] += 1
        elif winner_idx == 1:
            win_stats["O"] += 1
        else:
            win_stats["Tie"] += 1

        policy_losses = []
        entropy_losses = []

        total_steps = len(history)

        # backpropagation of rewards
        for step, (p_idx, log_prob, entropy) in enumerate(history):
            if winner_idx == -1:
                reward = 0.0
            else:
                # winner gets +1, loser -1: disconted by distance to result
                steps_to_end = total_steps - step - 1
                base_reward = 1.0 if p_idx == winner_idx else -1.0
                reward = base_reward * (gamma**steps_to_end)

            policy_losses.append(-log_prob * reward)
            entropy_losses.append(-entropy)

        # total loss (Loss = Policy Loss + Entropy Loss)
        optimizer.zero_grad()
        loss = (
            torch.stack(policy_losses).sum()
            + entropy_coef * torch.stack(entropy_losses).sum()
        )
        loss.backward()

        # grad clipping for stable train
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if episode % 250 == 0:
            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"Last 250 games -> X wins: {win_stats['X']}, O wins: {win_stats['O']}, Ties: {win_stats['Tie']} | "
                f"Loss: {loss.item():.4f}"
            )
            win_stats = {"X": 0, "O": 0, "Tie": 0}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model succesfully saved to: {save_path}")


if __name__ == "__main__":
    train(num_episodes=5000, rows=8, cols=8)
