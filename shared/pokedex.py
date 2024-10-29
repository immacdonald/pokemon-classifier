import json
import os


def get_pokedex():
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    input_path = "pokemon_list.json"

    with open(os.path.join(dir, input_path), mode="r") as data_file:
        pokedex = json.load(data_file)
        return pokedex
