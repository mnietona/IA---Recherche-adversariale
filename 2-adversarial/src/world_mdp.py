from dataclasses import dataclass
import lle
from lle import World, Action
from mdp import MDP, State


@dataclass
class MyState(State):
    ...


class WorldMDP(MDP[Action, MyState]):
    def __init__(self, world: World):
        self.world = world

    def reset(self):
        self.n_expanded_states = 0
        ...

    def available_actions(self, state: MyState) -> list[Action]:
        ...

    def is_final(self, state: MyState) -> bool:
        ...

    def transition(self, state: MyState, action: Action) -> MyState:
        ...


class BetterValueFunction(WorldMDP):
    def transition(self, state: MyState, action: Action) -> MyState:
        # Change the value of the state here.
        ...
