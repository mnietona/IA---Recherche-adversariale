import cv2
from lle import World
from world_mdp import WorldMDP
import random
from time import sleep

def visualize_world_mdp(world_filename: str):
    # Charger le monde à partir du fichier fourni
    w = World.from_file(world_filename)
    mdp = WorldMDP(w)
    
    state = mdp.reset()

    while not state.is_final():
        # Pour simplifier, nous allons simplement choisir une action aléatoire parmi les actions disponibles pour l'agent actif.
        actions = mdp.available_actions(state)
        chosen_action = random.choice(actions)
        
        # Transition vers le prochain état
        state = mdp.transition(state, chosen_action)
        
        # Visualiser l'état actuel du monde
        img = w.get_image()
        cv2.imshow("WorldMDP Visualization", img)
        # Attendre 500ms pour passer à l'étape suivante
        key = cv2.waitKey(500)
        if key == 27:  # Escape key
            break
    
    # Attendre 1 seconde puis fermer la fenêtre
    sleep(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Supposons que vous ayez une carte appelée "map.txt" que vous souhaitez visualiser
    visualize_world_mdp("level3")
