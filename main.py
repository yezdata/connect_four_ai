import os
import argparse
from typing import List, Dict, Callable

import game_engine as ge
from ai_strategy import STRATEGIES, monte_carlo_strategy

COLOR_RESET = "\033[0m"
COLOR_X = "\033[91m"  # Red
COLOR_O = "\033[94m"  # Blue
COLOR_GRID = "\033[90m"  # Dark gray
COLOR_NUMS = "\033[93m"  # Yellow


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def format_cell(cell: str) -> str:
    if cell == "X":
        return f"{COLOR_X}X{COLOR_RESET}"
    if cell == "O":
        return f"{COLOR_O}O{COLOR_RESET}"
    return " "


def human_move(state: List[List[str]]) -> int:
    cols = len(state[0])
    while True:
        raw_input = input(
            f"Which column do you want to play in? (0 to {cols - 1}): "
        ).strip()
        try:
            move = int(raw_input)
            if ge.valid_move(move, state):
                return move
            print(f"Invalid move. Column {move} is full or out of bounds.")
        except ValueError:
            print("Please enter a valid integer.")


def show_state(
    state: List[List[str]], last_action: str = "", current_status: str = ""
) -> None:
    clear_screen()
    print(last_action if last_action else "")
    print(current_status if current_status else "")
    print()

    cols = len(state[0])
    for row in state:
        row_str = " ".join(format_cell(cell) for cell in row)
        print(f"| {row_str} |")

    print(f"{COLOR_GRID}+" + "---" * cols + f"-+{COLOR_RESET}")
    indices_str = " ".join(f"{COLOR_NUMS}{i}{COLOR_RESET}" for i in range(cols))
    print(f"  {indices_str}  \n")


def tictactoe(
    rows: int = 8,
    cols: int = 8,
    human_starts: bool = True,
    ai_strategy_func: Callable[
        [List[List[str]], Dict[str, str]], int
    ] = monte_carlo_strategy,
) -> None:
    state = [[" " for _ in range(cols)] for _ in range(rows)]
    symbols = {
        "human": "X" if human_starts else "O",
        "pc": "O" if human_starts else "X",
    }

    current_player = "human" if human_starts else "pc"
    last_action = "Game begins."

    while not ge.is_tie(state):
        if current_player == "human":
            show_state(state, last_action, "Your turn...")
            h_move = human_move(state)
            state = ge.make_move(h_move, state, symbols["human"])
            last_action = (
                f"Player ({format_cell(symbols['human'])}) played in column {h_move}."
            )

            if ge.is_won(state):
                show_state(
                    state,
                    last_action,
                    f"Game over: Player ({format_cell(symbols['human'])}) won!",
                )
                return
            current_player = "pc"
        else:
            show_state(state, last_action, "Computer's turn...")
            pc_move = ai_strategy_func(state, symbols)
            state = ge.make_move(pc_move, state, symbols["pc"])
            last_action = (
                f"Computer ({format_cell(symbols['pc'])}) played in column {pc_move}."
            )

            if ge.is_won(state):
                show_state(
                    state,
                    last_action,
                    f"Game over: Computer ({format_cell(symbols['pc'])}) won!",
                )
                return
            current_player = "human"

    show_state(state, last_action, "Game over: Tie!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four / TicTacToe Game")
    parser.add_argument(
        "--ai",
        type=str,
        default="mc",
        choices=list(STRATEGIES.keys()),
        help="Select AI strategy: mc (Monte Carlo) or cnn (Reinforcement Learning)",
    )
    parser.add_argument("--rows", type=int, default=6, help="Board rows")
    parser.add_argument("--cols", type=int, default=6, help="Board columns")
    parser.add_argument(
        "--human-first", action="store_true", help="Set flag if human starts first"
    )

    args = parser.parse_args()

    selected_ai = STRATEGIES[args.ai]
    tictactoe(
        rows=args.rows,
        cols=args.cols,
        human_starts=args.human_first,
        ai_strategy_func=selected_ai,
    )
