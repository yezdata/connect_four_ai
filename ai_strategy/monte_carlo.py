from typing import List, Dict
from random import randint
import game_engine as ge


def random_strategy(state: List[List[str]]) -> int:
    valid_moves = [move for move in range(len(state[0])) if ge.valid_move(move, state)]
    return valid_moves[randint(0, len(valid_moves) - 1)]


def one_sim(state: List[List[str]], move: int, symbols: Dict[str, str]) -> int:
    sim_state = [row.copy() for row in state]
    sim_state = ge.make_move(move, sim_state, symbols["pc"])

    if ge.is_won(sim_state):
        return 1

    current_player = "human"

    while not ge.is_tie(sim_state):
        if current_player == "human":
            h_move = random_strategy(sim_state)
            sim_state = ge.make_move(h_move, sim_state, symbols["human"])
            if ge.is_won(sim_state):
                return -1
        else:
            pc_move = random_strategy(sim_state)
            sim_state = ge.make_move(pc_move, sim_state, symbols["pc"])
            if ge.is_won(sim_state):
                return 1

        current_player = "pc" if current_player == "human" else "human"

    return 0


def monte_carlo_sim(
    state: List[List[str]], move: int, symbols: Dict[str, str], sim_count: int = 200
) -> float:
    wins = [one_sim(state, move, symbols) for _ in range(sim_count)]
    return sum(wins) / len(wins)


def monte_carlo_strategy(state: List[List[str]], symbols: Dict[str, str]) -> int:
    """Main decision func for the Monte Carlo strategy."""
    candidate_moves = [
        move for move in range(len(state[0])) if ge.valid_move(move, state)
    ]
    exp_results = [monte_carlo_sim(state, move, symbols) for move in candidate_moves]
    best_index = max(range(len(exp_results)), key=lambda i: exp_results[i])
    return candidate_moves[best_index]
