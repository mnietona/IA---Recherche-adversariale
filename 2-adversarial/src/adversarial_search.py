from mdp import MDP, S, A
from typing import Tuple

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """ Retourne la meilleure action selon l'algorithme minimax."""
    if state.current_agent != 0:
        raise ValueError("The state should be for agent 0 to play.")

    # Fonction pour maximiser la valeur et l'action
    def _max(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('-inf')
        best_action = None
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            # Vérifie si l'état suivant a un type d'agent différent de l'état actuel
            decrease_depth = 1 if next_state.current_agent != state.current_agent else 0
            next_state_value, _ = _min(next_state, depth + decrease_depth)
            if next_state_value > value:
                value, best_action = next_state_value, action

        return value, best_action

    # Fonction pour minimiser la valeur pour les adversaires
    def _min(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('inf')
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            decrease_depth = 1 if next_state.current_agent == 0 else 0
            if next_state.current_agent == 0:  # si c'est le tour de l'agent 0
                next_state_value, _ = _max(next_state, depth + decrease_depth)
            else:  # si c'est le tour d'un autre agent
                next_state_value, _ = _min(next_state, depth + decrease_depth)
            value = min(value, next_state_value)

        return value, None

    _, best_action = _max(state, 0)
    return best_action

def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """ Retourne la meilleure action selon l'algorithme alpha-beta. """
    if state.current_agent != 0:
        raise ValueError("The state should be for agent 0 to play.")
    
    def _max(state: S, depth: int, alpha: float, beta: float) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('-inf')
        best_action = None
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            decrease_depth = 1 if next_state.current_agent != state.current_agent else 0
            next_state_value, _ = _min(next_state, depth + decrease_depth, alpha, beta)
            if next_state_value > value:
                value, best_action = next_state_value, action
            # Élagage alpha
            if value >= beta:
                return value, best_action
            alpha = max(alpha, value)
        return value, best_action

    def _min(state: S, depth: int, alpha: float, beta: float) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('inf')
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            decrease_depth = 1 if next_state.current_agent == 0 else 0
            if next_state.current_agent == 0:
                next_state_value, _ = _max(next_state, depth + decrease_depth, alpha, beta)
            else:
                next_state_value, _ = _min(next_state, depth + decrease_depth, alpha, beta)
            value = min(value, next_state_value)
            # Élagage beta
            if value <= alpha:
                return value, None
            beta = min(beta, value)
        return value, None

    _, best_action = _max(state, 0, float('-inf'), float('inf'))
    return best_action

def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """ Retourne la meilleure action selon l'algorithme expectimax. """

    if state.current_agent != 0:
        raise ValueError("The state should be for agent 0 to play.")
    
    def _max(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = float('-inf')
        best_action = None
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            decrease_depth = 1 if next_state.current_agent != state.current_agent else 0
            next_state_value, _ = _expectation(next_state, depth + decrease_depth)
            if next_state_value > value: 
                value, best_action = next_state_value, action
        return value, best_action

    def _expectation(state: S, depth: int) -> Tuple[float, A]:
        if mdp.is_final(state) or depth == max_depth:
            return state.value, None
        value = 0
        actions = mdp.available_actions(state)
        prob = 1.0 / len(actions)
        for action in actions:
            next_state = mdp.transition(state, action)
            decrease_depth = 1 if next_state.current_agent == 0 else 0
            if next_state.current_agent == 0: 
                next_state_value, _ = _max(next_state, depth + decrease_depth)
            else:
                next_state_value, _ = _expectation(next_state, depth + decrease_depth)
            value += prob * next_state_value 
        return value, None

    _, best_action = _max(state, 0)
    return best_action
