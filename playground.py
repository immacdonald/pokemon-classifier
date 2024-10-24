import json
import os
from pprint import pprint


def count_types(pokedex):
    types = {}
    for pokemon in pokedex:
        type = pokemon["primary_type"]
        # type = pokemon['secondary_type']
        if type in types:
            types[type] += 1
        else:
            types[type] = 1

    return types


def main() -> None:
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    input_path = "pokemon_list.json"

    with open(os.path.join(dir, input_path), mode="r") as data_file:
        pokedex = json.load(data_file)
        pprint(count_types(pokedex))


if __name__ == "__main__":
    main()
