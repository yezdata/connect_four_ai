import os
from typing import Dict, List, Optional
import torch
from torch.distributions import Categorical

import game_engine as ge
from rl_cnn.model import PolicyCNN, state_to_tensor

# Cached global instance to avoid reloading model weights on every move
_MODEL: Optional[PolicyCNN] = None
_DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rl_cnn",
    "connect4_policy.pt",
)


def load_model(
    weights_path: str = _DEFAULT_WEIGHTS_PATH, rows: int = 6, cols: int = 6
) -> PolicyCNN:
    """Loads and caches the PolicyCNN model in evaluation mode."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Trained model weights not found at: {weights_path}\n"
            )
        model = PolicyCNN(rows=rows, cols=cols).to(_DEVICE)
        state_dict = torch.load(weights_path, map_location=_DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        _MODEL = model
    return _MODEL


def sample_action(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> int:
    """Selects an action index using argmax, temperature scaling, and/or top-k sampling."""
    # Deterministic selection when temperature is effectively zero
    if temperature <= 1e-5:
        return int(torch.argmax(logits).item())

    # Scale logits by temperature: P(a_i) = exp(z_i / T) / sum(exp(z_j / T))
    scaled_logits = logits / temperature

    # Apply top-k filtering if requested
    if top_k is not None and 0 < top_k < scaled_logits.size(-1):
        top_values, _ = torch.topk(scaled_logits, k=top_k)
        min_allowed_value = top_values[-1]
        # Mask out everything strictly lower than the k-th highest logit
        scaled_logits = torch.where(
            scaled_logits >= min_allowed_value,
            scaled_logits,
            torch.tensor(-1e9, device=scaled_logits.device),
        )

    # Sample from the resulting categorical distribution
    dist = Categorical(logits=scaled_logits)
    return int(dist.sample().item())


def rl_cnn_strategy(
    state: List[List[str]],
    symbols: Dict[str, str],
    weights_path: str = _DEFAULT_WEIGHTS_PATH,
    temperature: float = 0.2,
    top_k: Optional[int] = None,
) -> int:
    """Decision strategy function matching the Connect Four engine interface."""
    rows = len(state)
    cols = len(state[0])
    model = load_model(weights_path=weights_path, rows=rows, cols=cols)

    current_sym = symbols["pc"]
    opponent_sym = symbols["human"]

    with torch.no_grad():
        # Encode board state into channel tensor (1, 2, rows, cols)
        s_tensor = state_to_tensor(state, current_sym, opponent_sym, device=_DEVICE)

        # Build valid moves boolean mask (1, cols)
        valid_moves = [ge.valid_move(c, state) for c in range(cols)]
        mask_tensor = torch.tensor([valid_moves], dtype=torch.bool, device=_DEVICE)

        # Forward pass returning shape (cols,)
        logits = model(s_tensor, valid_mask=mask_tensor).squeeze(0)

        # Pick column using specified inference strategy
        chosen_col = sample_action(logits, temperature=temperature, top_k=top_k)

    # Sanity guard: fallback to first valid move if an illegal move is somehow sampled
    if not ge.valid_move(chosen_col, state):
        fallback_candidates = [c for c in range(cols) if ge.valid_move(c, state)]
        chosen_col = fallback_candidates[0]

    return chosen_col
