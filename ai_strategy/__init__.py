from ai_strategy.monte_carlo import monte_carlo_strategy
from ai_strategy.rl_cnn import rl_cnn_strategy

STRATEGIES = {
    "mc": monte_carlo_strategy,
    "cnn": rl_cnn_strategy,
}
