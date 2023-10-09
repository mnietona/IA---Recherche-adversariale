# Variables pour les commandes fréquemment utilisées
PYTHON = python3


all: main

main:
	$(PYTHON) src/main.py

test:
	pytest tests/graph_mdp.py tests/test_alpha_beta.py tests/test_expectimax.py tests/test_mdp.py tests/test_minimax.py 

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

t:
	pytest tests/test_mdp.py

.PHONY: all main test clean
