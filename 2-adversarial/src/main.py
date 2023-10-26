from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta
import csv
import cv2


WORLDS = [
World("""
      . . . S2 S0 S1 .
      G . . . . . G
      L0E . . . . . .
      . . @ G . . .
      G . . . . . G
      . . . . @ G .
      . . X . . @ G
      G X X . . . .
      """
),

World("""
      . . . S2 S0 S1 .
      G . . . . . G
      @ . . @ @ @ @
      . . G G . . .
      G . . . . @ G
      . . . . @ G .
      . . . . . @ G
      G X X X . . .
      """
),

World("""
      . S0 S1 S2 . . .
      . . . . . . G
      L1E . . . @ . .
      . G . . G . G
      . . @ . . . L2W
      . . . . G . @
      . . @ . . . .
      . G . X X X G
      """
)

]


WMDPS = (WorldMDP,BetterValueFunction)

ALGOS = ((minimax, "minimax"), (alpha_beta, "alpha_beta"))


def main():
    results = []
    for i in range(1, len(WORLDS)):
        cv2.imwrite(f"world_{i}.png", WORLDS[i].get_image())
        for depth in range(1, 10):
            print(depth)
            for WMDP in WMDPS:
                for algo, name in ALGOS:
                    if BetterValueFunction == WMDP and algo == minimax:continue

                    world = WMDP(WORLDS[i])
                    s0 = world.reset()

                    action = algo(world, s0, depth)
                    n_states = world.n_expanded_states
                    results.append([i, depth, WMDP.__name__, name, action, n_states])

    # Écrivez les résultats dans un fichier CSV
    with open('results_newworld1.csv', 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Acttion','Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'World': str(result[0]), 'Depth': result[1], 'WMDP': result[2], 'Algorithm': result[3], 'Action': result[4], 'Expanded States': result[5]})

    print("Les résultats ont été enregistrés dans le fichier results.csv")

    # Transformation des résultats pour l'affichage souhaité
    organized_results = {}
    for r in results:
        algo_name = r[3]
        # Modifier le nom de l'algorithme si BetterValueFunction est utilisé
        if r[2] == "BetterValueFunction":
            algo_name += " (BetterValueFunction)"
        key = (r[0], algo_name)  # (World, Algorithm)
        value = (r[1], r[5])  # (Depth, Expanded States)
        if key not in organized_results:
            organized_results[key] = []
        organized_results[key].append(value)

    # Affichage des résultats
    for key, values in organized_results.items():
        values.sort(key=lambda x: int(x[0]))
        coords = " ".join(["({},{})".format(v[0], v[1]) for v in values])
        print("Monde {} - Algo : {}".format(key[0], key[1]))
        print(coords)
        print("\n")


if __name__ == "__main__":
    main()