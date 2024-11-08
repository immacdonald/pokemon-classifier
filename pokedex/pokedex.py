import json
import os


def get_pokedex(standardize=False):
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    input_path = "pokemon_list.json"

    with open(os.path.join(dir, input_path), mode="r") as data_file:
        pokedex = json.load(data_file)
        if standardize:
            pokedex = [pokemon for pokemon in pokedex if not (pokemon.get("mega") or pokemon.get("region"))]

        return pokedex


# Organized based on the internal type ID used in Generation IX
def get_types():
    return ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark", "fairy"]
