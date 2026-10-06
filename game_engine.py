from typing import List


def is_won(state: List[List[str]]) -> bool:
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


def is_tie(state: List[List[str]]) -> bool:
    return all(cell != " " for row in state for cell in row)


def valid_move(move: int, state: List[List[str]]) -> bool:
    return 0 <= move < len(state[0]) and state[0][move] == " "


def make_move(move: int, state: List[List[str]], symbol: str) -> List[List[str]]:
    for i in range(len(state) - 1, -1, -1):
        if state[i][move] == " ":
            state[i][move] = symbol
            return state
    return state
