from lle import Action
from mdp import MDP, S, A
from typing import Tuple

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """ Return the best action according to the minimax algorithm."""
    if state.current_agent != 0:
        raise ValueError("The state should be for agent 0 to play.")

    
    # Wrapper to maximize value and action
    def _max(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('-inf')
        best_action = None
        for action in mdp.available_actions(state):
            child = mdp.transition(state, action)
            # Check if next state has a different agent type (0 vs non-0)
            decrease_depth = 1 if child.current_agent != state.current_agent else 0
            child_value, _ = _min(child, depth + decrease_depth)
            if child_value > value:
                value, best_action = child_value, action

        return value, best_action

    # Wrapper to minimize value for the adversaries
    def _min(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('inf')
        for action in mdp.available_actions(state):
            child = mdp.transition(state, action)
            # Check if next state has a different agent type (0 vs non-0)
            decrease_depth = 1 if child.current_agent == 0 else 0
            if child.current_agent == 0:  # if it's agent 0's turn
                child_value, _ = _max(child, depth + decrease_depth)
            else:  # if it's another agent's turn
                child_value, _ = _min(child, depth + decrease_depth)
            value = min(value, child_value)

        return value, None

    _, best_action = _max(state, 0)
    return best_action


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    ...
