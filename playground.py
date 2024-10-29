from shared.pokedex import get_pokedex
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
    pokedex = get_pokedex()
    print(len(pokedex))
    #pprint(count_types(pokedex))


if __name__ == "__main__":
    main()
