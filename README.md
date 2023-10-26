
# Projet d'Intelligence Artificielle - Recherche adversariale


## Introduction
Ce projet d'Intelligence Artificielle se focalise sur les techniques de recherche adversariale appliquées aux graphes. Contrairement à un environnement coopératif comme celui de LLE, ce projet transforme le cadre en un Processus Décisionnel de Markov compétitif où un seul agent est actif. L'objectif principal est d'optimiser la collecte de gemmes par l'agent 0, en considérant tous les autres agents comme des adversaires potentiels.

## Prérequis
- Python 3.10
- Poetry (pour l'installation des dépendances)

## Installation
1. Clonez ce dépôt sur votre machine locale.
2. Installez Poetry en suivant les instructions sur [le site officiel de Poetry](https://python-poetry.org/docs/#installation).
3. Ouvrez un terminal dans le répertoire du projet.

## Configuration de l'environnement
L'environnement de projet "Laser Learning Environment" est décrit comme suit :
- Entre 1 et 4 agents se déplacent et collectent des gemmes sur une grille.
- Des lasers de couleur bloquent le passage aux agents d'une autre couleur, mais un agent de la même couleur peut bloquer le rayon laser pour permettre à d'autres agents de passer.
- Les cases de départ sont représentées par des carrés de la couleur de l'agent qui commence (en haut, au milieu).
- Les cases de sortie sont indiquées par des cases encadrées en noir (en bas à droite).
- Le jeu est terminé quand tous les agents ont atteint la sortie.

## Structure du Projet
- `src/`: Le répertoire source contient toutes les implémentations des classes de problème et des algorithmes de recherche.
- `tests/`: Les tests unitaires pour les classes de problème et les algorithmes de recherche.
- `src/main.py`: Un script pour exécuter les algorithmes de recherche sur les problèmes spécifiques.

## Utilisation
Pour exécuter le projet, ouvrez un terminal dans le répertoire du projet et utilisez les commandes suivantes :

- Pour exécuter le projet:
  ```shell
  poetry shell
  poetry install
  make
  ```

- Pour exécuter les tests unitaires:
  ```shell
  make test
  ```

