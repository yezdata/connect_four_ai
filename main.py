import argparse
from typing import Callable

import game_engine as ge
from ai_strategy import STRATEGIES, monte_carlo_strategy


def tictactoe(
    rows: int,
    cols: int,
    human_starts: bool = False,
    ai_strategy_func: Callable[
        [list[list[str]], dict[str, str]], int
    ] = monte_carlo_strategy,
) -> None:
    state = [[" " for _ in range(cols)] for _ in range(rows)]
    symbols = {
        "human": "X" if human_starts else "O",
        "ai": "O" if human_starts else "X",
    }

    current_player = "human" if human_starts else "ai"
    last_action = "Game begins."

    while not ge.is_tie(state):
        if current_player == "human":
            ge.show_state(state, last_action, "Your turn...")
            h_move = ge.human_move(state)
            state = ge.make_move(h_move, state, symbols["human"])
            last_action = f"Player ({ge.format_cell(symbols['human'])}) played in column {h_move}."

            if ge.is_won(state):
                ge.show_state(
                    state,
                    last_action,
                    f"Game over: Player ({ge.format_cell(symbols['human'])}) won!",
                )
                return
            current_player = "ai"
        else:
            ge.show_state(state, last_action, "AI's turn...")
            ai_move = ai_strategy_func(state, symbols)
            state = ge.make_move(ai_move, state, symbols["ai"])
            last_action = (
                f"AI ({ge.format_cell(symbols['ai'])}) played in column {ai_move}."
            )

            if ge.is_won(state):
                ge.show_state(
                    state,
                    last_action,
                    f"Game over: AI ({ge.format_cell(symbols['ai'])}) won!",
                )
                return
            current_player = "human"

    ge.show_state(state, last_action, "Game over: Tie!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four / TicTacToe Game")
    parser.add_argument(
        "--ai",
        type=str,
        default="mc",
        choices=list(STRATEGIES.keys()),
        help="Select AI strategy: mc (Monte Carlo) or cnn (Reinforcement Learning)",
    )
    parser.add_argument("--rows", type=int, default=8, help="Board rows")
    parser.add_argument("--cols", type=int, default=8, help="Board columns")
    parser.add_argument(
        "--human-first", action="store_true", help="Set flag if human starts first"
    )

    args = parser.parse_args()

    if args.ai == "cnn" and (args.rows != 8 or args.cols != 8):
        raise ValueError(
            "The CNN strategy is only supported for a 8x8 board. Please use --rows 8 --cols 8."
        )

    selected_ai = STRATEGIES[args.ai]
    tictactoe(
        rows=args.rows,
        cols=args.cols,
        human_starts=args.human_first,
        ai_strategy_func=selected_ai,
    )
