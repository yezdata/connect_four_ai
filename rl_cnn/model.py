import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def state_to_tensor(
    state: List[List[str]],
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


class PolicyCNN(nn.Module):
    def __init__(self, rows: int = 8, cols: int = 8, kernel_size: int = 3):
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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
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
            logits = torch.where(
                valid_mask, logits, torch.tensor(-1e4, device=logits.device)
            )

        return logits
