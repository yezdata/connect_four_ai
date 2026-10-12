import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connection (residual)
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return F.relu(out)


class ActorCriticCNN(nn.Module):
    def __init__(
        self,
        rows: int,
        cols: int,
        kernel_size: int = 3,
        num_blocks: int = 10,
        channels: int = 128,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        pad = kernel_size // 2

        self.init_conv = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=kernel_size, padding=pad),
            nn.GroupNorm(1, channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(channels, kernel_size) for _ in range(num_blocks)]
        )

        # Policy Head (Actor): chooses moves
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),  # bottleneck
            nn.GroupNorm(1, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * rows * cols, cols),
        )

        # Value Head (Critic): evaluates state of board
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),  # bottleneck
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * rows * cols, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None):
        h = self.init_conv(x)
        for block in self.res_blocks:
            h = block(h)

        logits = self.policy_head(h)
        value = self.value_head(h)

        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e4)

        return logits, value


class PolicyCNN(nn.Module):
    def __init__(self, rows: int, cols: int, kernel_size: int = 3):
        super().__init__()
        self.rows = rows
        self.cols = cols

        # receptive field 9x9
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=64, kernel_size=kernel_size, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1
        )

        # MLP part
        self.fc1 = nn.Linear(64 * rows * cols, 64)
        self.fc_out = nn.Linear(64, cols)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: tensor (batch, 2, rows, cols)
        valid_mask: bool tensor (batch, cols), where True = valid col
        returns logits for every col
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = torch.flatten(h, start_dim=1)
        h = F.relu(self.fc1(h))
        logits = self.fc_out(h)

        if valid_mask is not None:
            # non-valid moves are set to -inf -> after softmax 0 prob
            logits = logits.masked_fill(~valid_mask, -1e4)

        return logits


def state_to_tensor(
    state: list[list[str]],
    current_symbol: str,
    opponent_symbol: str,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Converts state of the board to tensor (1, 2, rows, cols):
        Channel 0: positions of the current Player
        Channel 1: positions of opponent
    """
    rows = len(state)
    cols = len(state[0])
    tensor = torch.zeros((1, 2, rows, cols), dtype=torch.float32, device=device)

    for r in range(rows):
        for c in range(cols):
            cell = state[r][c]
            if cell == current_symbol:
                tensor[0, 0, r, c] = 1.0
            elif cell == opponent_symbol:
                tensor[0, 1, r, c] = 1.0

    return tensor
