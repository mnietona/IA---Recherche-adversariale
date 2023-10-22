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
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        new_state = super().transition(state, action) 
        
        # Encourage la collecte de gemmes
        gem_bonus = sum(state.world_state.gems_collected) * 1000
        
        # On penalise le mouvement pour éviter les mouvements excessifs
        movement_penalty = 0 if action == "Stay" else -1
        
        new_value = gem_bonus + movement_penalty
        new_state.value = new_value
        
        return new_state

