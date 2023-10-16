
from adversarial_search import minimax
from lle import World
from world_mdp import BetterValueFunction, WorldMDP



def test_better_value_function_expansion1():
    world = World(
        """
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
"""
    )
    world_mdp = WorldMDP(world)
    world_better = BetterValueFunction(world)

    minimax(world_mdp, world_mdp.reset(), 3)
    minimax(world_better, world_better.reset(), 3)

    print(world_mdp.n_expanded_states)
    print(world_better.n_expanded_states)


    assert world_better.n_expanded_states < world_mdp.n_expanded_states

def test_better_value_function_expansion2():
    world = World(
        """
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
"""
    )
    world_mdp = WorldMDP(world)
    world_better = BetterValueFunction(world)

    minimax(world_mdp, world_mdp.reset(), 4)
    minimax(world_better, world_better.reset(), 4)

    print(world_mdp.n_expanded_states)
    print(world_better.n_expanded_states)


    assert world_better.n_expanded_states < world_mdp.n_expanded_states


def test_better_value_function_expansion3():
    world = World(
        """
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
    )
    world_mdp = WorldMDP(world)
    world_better = BetterValueFunction(world)

    minimax(world_mdp, world_mdp.reset(), 3)
    minimax(world_better, world_better.reset(), 3)

    print(world_mdp.n_expanded_states)
    print(world_better.n_expanded_states)

    assert world_better.n_expanded_states < world_mdp.n_expanded_states


def test_better_value_function_expansion4():
    world = World(
        """
        S0 . G G .
        G  @ @ @ .
        .  . X X X
        S1 . . . S2
"""
    )
    world_mdp = WorldMDP(world)
    world_better = BetterValueFunction(world)

    minimax(world_mdp, world_mdp.reset(), 3)
    minimax(world_better, world_better.reset(), 3)

    print(world_mdp.n_expanded_states)
    print(world_better.n_expanded_states)

    assert world_better.n_expanded_states < world_mdp.n_expanded_states
