format:
	isort . && black . && flake8

pokedex:
	python3 -m pokedex.compile_pokedex

train:
	python3 -m training.training

.PHONY: format pokedex train