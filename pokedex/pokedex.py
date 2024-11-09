import json
import os

from .pokemon import Pokemon, PokemonData, validate_pokemon


def get_pokedex_data(standardize=False) -> list[PokemonData]:
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    input_path = "pokemon_list.json"

    with open(os.path.join(dir, input_path), mode="r") as data_file:
        pokedex = json.load(data_file)
        pokedex = validate_pokemon(pokedex)

        if standardize:
            pokedex = [pokemon for pokemon in pokedex if not (pokemon.get("mega") or pokemon.get("region"))]

        return pokedex


def get_pokedex(standardize=False) -> list[Pokemon]:
    pokedex = get_pokedex_data(standardize)
    pokemon_list = []

    for pokemon in pokedex:
        pokemon_list.append(
            Pokemon(
                name=pokemon.get("name"),
                number=pokemon.get("number"),
                primary_type=pokemon.get("primary_type"),
                secondary_type=pokemon.get("secondary_type"),
                generation=pokemon.get("generation"),
                region=pokemon.get("region"),
                mega=pokemon.get("mega"),
                form=pokemon.get("form"),
                is_alternate=pokemon.get("is_alternate"),
                alternate_count=pokemon.get("alternate_count"),
                variants=pokemon.get("variants", []),
            )
        )

    return pokemon_list


# Organized based on the internal type ID used in Generation IX
def get_types():
    return ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark", "fairy"]
