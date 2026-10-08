import os


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


def human_move(state: list[list[str]]) -> int:
    cols = len(state[0])
    while True:
        raw_input = input(
            f"Which column do you want to play in? (0 to {cols - 1}): "
        ).strip()
        try:
            move = int(raw_input)
            if valid_move(move, state):
                return move
            print(f"Invalid move. Column {move} is full or out of bounds.")
        except ValueError:
            print("Please enter a valid integer.")


def show_state(
    state: list[list[str]],
    last_action: str = "",
    current_status: str = "",
    is_training: bool = False,
) -> None:
    if not is_training:
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


def is_won(state: list[list[str]]) -> bool:
    rows = len(state)
    cols = len(state[0])

    for i in range(rows):
        for j in range(cols):
            if j <= cols - 4:
                test_row_l = state[i][j : (j + 4)]
                if len(set(test_row_l)) == 1 and test_row_l[0] != " ":
                    return True

            if i <= rows - 4:
                test_col_l = [state[i + k][j] for k in range(4)]
                if len(set(test_col_l)) == 1 and test_col_l[0] != " ":
                    return True

            if i <= rows - 4 and j <= cols - 4:
                test_diag_right_l = [state[i + k][j + k] for k in range(4)]
                if len(set(test_diag_right_l)) == 1 and test_diag_right_l[0] != " ":
                    return True

            if i >= 3 and j <= cols - 4:
                test_diag_left_l = [state[i - k][j + k] for k in range(4)]
                if len(set(test_diag_left_l)) == 1 and test_diag_left_l[0] != " ":
                    return True

    return False


def is_tie(state: list[list[str]]) -> bool:
    return all(cell != " " for row in state for cell in row)


def valid_move(move: int, state: list[list[str]]) -> bool:
    return 0 <= move < len(state[0]) and state[0][move] == " "


def make_move(move: int, state: list[list[str]], symbol: str) -> list[list[str]]:
    for i in range(len(state) - 1, -1, -1):
        if state[i][move] == " ":
            state[i][move] = symbol
            return state
    return state
