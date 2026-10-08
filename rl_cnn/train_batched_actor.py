import os
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import game_engine as ge
from rl_cnn.model import ActorCriticCNN


class BatchedConnectFourEnv:
    def __init__(self, num_envs: int, rows: int, cols: int):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.symbols = ["X", "O"]

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
        self.histories: list[list[tuple[torch.Tensor, torch.Tensor, int, int]]] = [
            [] for _ in range(self.num_envs)
        ]

    def reset_single(self, env_idx: int):
        self.boards[env_idx] = [
            [" " for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.turn_idx[env_idx] = 0
        self.histories[env_idx] = []

    def is_random_turn(self, env_idx: int) -> bool:
        etype = self.env_types[env_idx]
        cur_turn = self.turn_idx[env_idx]
        if etype == 1 and cur_turn == 1:
            return True
        if etype == 2 and cur_turn == 0:
            return True
        return False

    def get_batched_states_and_masks(
        self, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.zeros(
            (self.num_envs, 2, self.rows, self.cols), dtype=torch.float32, device=device
        )
        masks = torch.zeros((self.num_envs, self.cols), dtype=torch.bool, device=device)

        for i in range(self.num_envs):
            board = self.boards[i]
            cur_sym = self.symbols[self.turn_idx[i]]
            opp_sym = self.symbols[1 - self.turn_idx[i]]

            for r in range(self.rows):
                for c in range(self.cols):
                    cell = board[r][c]
                    if cell == cur_sym:
                        states[i, 0, r, c] = 1.0
                    elif cell == opp_sym:
                        states[i, 1, r, c] = 1.0

            for c in range(self.cols):
                if ge.valid_move(c, board):
                    masks[i, c] = True

        return states, masks


def train_batched(
    total_completed_games: int,
    num_envs: int,
    batch_update_games: int,
    rows: int,
    cols: int,
    checkpoint_interval: int = 400,
    lr: float = 1e-4,
    gamma: float = 0.95,
    entropy_coef: float = 0.03,
    value_coef: float = 0.5,
    save_path: str = "rl_cnn/connect4_actor",
):
    save_path = f"{save_path}_r{rows}_c{cols}.pt"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Training on device: {device} | Parallel environments: {num_envs}")

    model = ActorCriticCNN(rows=rows, cols=cols, num_blocks=10, channels=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    env = BatchedConnectFourEnv(num_envs=num_envs, rows=rows, cols=cols)

    total_updates = total_completed_games // batch_update_games

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6
    )

    completed_games_count = 0
    updates_count = 0
    stats = {"X": 0, "O": 0, "Tie": 0}
    finished_trajectories = []

    while completed_games_count < total_completed_games:
        # rollout
        model.eval()
        with torch.no_grad():
            states, masks = env.get_batched_states_and_masks(device)
            logits, _ = model(states, valid_mask=masks)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        # state step
        for i in range(num_envs):
            cur_turn = env.turn_idx[i]
            board = env.boards[i]
            is_random = env.is_random_turn(i)

            if is_random:
                valid_cols = torch.where(masks[i])[0].tolist()
                act = random.choice(valid_cols)
            else:
                act = actions[i].item()
                env.histories[i].append(
                    (states[i].cpu(), masks[i].cpu(), act, cur_turn)
                )

            ge.make_move(act, board, env.symbols[cur_turn])

            if ge.is_won(board):
                winner = cur_turn
                stats["X" if winner == 0 else "O"] += 1
                if env.histories[i]:
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
                env.turn_idx[i] = 1 - env.turn_idx[i]

        # training update
        if len(finished_trajectories) >= batch_update_games:
            b_states, b_masks, b_actions, b_returns = [], [], [], []

            for history, winner in finished_trajectories:
                n_steps = len(history)
                for step, (s, m, a, p_idx) in enumerate(history):
                    if winner == -1:
                        ret = 0.0
                    else:
                        steps_to_end = n_steps - step - 1
                        base_reward = 1.0 if p_idx == winner else -1.0
                        ret = base_reward * (gamma**steps_to_end)

                    b_states.append(s)
                    b_masks.append(m)
                    b_actions.append(a)
                    b_returns.append(ret)

            # create train tensors
            b_states = torch.stack(b_states).to(device)
            b_masks = torch.stack(b_masks).to(device)
            b_actions = torch.tensor(b_actions, dtype=torch.long, device=device)
            b_returns = torch.tensor(b_returns, dtype=torch.float32, device=device)

            model.train()
            optimizer.zero_grad()

            logits, values = model(b_states, valid_mask=b_masks)
            dist = Categorical(logits=logits)

            log_probs = dist.log_prob(b_actions)
            entropies = dist.entropy()

            advantages = b_returns - values.detach().squeeze(-1)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values.squeeze(-1), b_returns)
            entropy_loss = -entropies.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Flush off-policy data - drop old weights results
            for i in range(num_envs):
                env.histories[i].clear()
            finished_trajectories.clear()

            updates_count += 1

            if updates_count % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Games: {completed_games_count:6d}/{total_completed_games} | \n"
                    f"LR: {current_lr:.2e} | \n"
                    f"Wins -> X: {stats['X']}, O: {stats['O']}, Ties: {stats['Tie']} | \n"
                    f"Loss: {loss.item():.4f} (Pol: {policy_loss.item():.4f}, Val: {value_loss.item():.4f}, Ent: {entropies.mean().item():.4f})"
                )
                stats = {"X": 0, "O": 0, "Tie": 0}

                # ge.show_state(
                #     env.boards[0],
                #     last_action="Sample board state after training update.",
                #     current_status="",
                #     is_training=True,
                # )

            if updates_count % checkpoint_interval == 0:
                base_name = os.path.basename(save_path).replace(
                    ".pt", f"_ckpt_{completed_games_count}.pt"
                )
                checkpoint_dir = "checkpoints"
                checkpoint_path = os.path.join(checkpoint_dir, base_name)

                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "games": completed_games_count,
                        "updates": updates_count,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                    },
                    checkpoint_path,
                )
                print(f"[*] Checkpoint saved: {checkpoint_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train_batched(
        total_completed_games=10_000_000,
        num_envs=1024,
        batch_update_games=128,
        checkpoint_interval=400,
        rows=8,
        cols=8,
        lr=1e-4,
        gamma=0.95,
        entropy_coef=0.03,
        value_coef=0.5,
    )
