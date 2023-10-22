from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State
from typing import List, Tuple

@dataclass
class MyWorldState(State):
    world: World
    world_state: WorldState
    current_agent: int
    value: float

    def is_final(self):
        return self.world.done
    
    
class WorldMDP(MDP[MyWorldState, Action]):
    def __init__(self, world: World):
        self.world = world

    def reset(self) -> MyWorldState:
        self.n_expanded_states = 0
        self.world.reset()
        return MyWorldState(world=self.world, world_state=self.world.get_state(), current_agent=0, value=0)

    def available_actions(self, state: MyWorldState) -> List[Action]:
        actions_for_all_agents = self.world.available_actions()
        return actions_for_all_agents[state.current_agent]

    def is_final(self, state: MyWorldState) -> bool:
        return state.is_final()
    
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        self.world.set_state(state.world_state)
        
        if self.is_final(state): return state
        
        self.n_expanded_states += 1
        actions_for_all_agents = [Action.STAY] * self.world.n_agents
        actions_for_all_agents[state.current_agent] = action
        reward = self.world.step(actions_for_all_agents)
        
        if state.current_agent == 0: 
            if self.world.agents[0].is_dead:
                new_value = lle.REWARD_AGENT_DIED
            else:
                new_value = state.value + reward
        else:
            new_value = state.value

        next_agent = (state.current_agent + 1) % self.world.n_agents
        
        new_state = MyWorldState(world=self.world, world_state=self.world.get_state(), current_agent=next_agent, value=new_value)
        return new_state


class BetterValueFunction(WorldMDP):
    
    def manhattan_distance(self, start: Tuple[int, int], target: Tuple[int, int]) -> int:
        """Calcule la distance de Manhattan entre deux points."""
        return abs(start[0] - target[0]) + abs(start[1] - target[1])

    def min_manhattan_distance(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> int:
        """Renvoie la distance min de Manhattan. """
        return min(self.manhattan_distance(start, target) for target in targets)

    def compute_heuristic(self, state: MyWorldState) -> float:
        """Calcule la valeur heuristique basée sur la distance aux objets et le nombre de gemmes collectées."""

        # Position actuelle de l'agent
        agent_position = state.world_state.agents_positions[state.current_agent]

        # Distance à la gemme la plus proche
        uncollected_gems_positions = [gem[0] for gem, collected in zip(state.world.gems, state.world_state.gems_collected) if not collected]
        
        if uncollected_gems_positions:  # Vérifier s'il reste des gemmes à collecter
            dist_gem = self.min_manhattan_distance(agent_position, uncollected_gems_positions)
        else:
            dist_gem = 0  # Toutes les gemmes ont été collectées

        # Distance à la sortie la plus proche
        dist_exit = self.min_manhattan_distance(agent_position, state.world.exit_pos)

        heuristic_value = -(state.world.gems_collected * 10 + dist_gem + dist_exit)

        return heuristic_value


    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        new_state = super().transition(state, action)  
        
        heuristic_value = self.compute_heuristic(new_state)
        new_value = new_state.value + heuristic_value
        
        new_state.value = new_value
        
        return new_state
