import json
import os

from .pokemon import Pokemon, PokemonData, validate_pokemon


def get_pokedex_data(standardize=False, start_at=0, end_at=2000) -> list[PokemonData]:
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    input_path = "pokemon_list.json"

    with open(os.path.join(dir, input_path), mode="r") as data_file:
        data = json.load(data_file)
        pokedex: list[PokemonData] = validate_pokemon(data)

        pokedex = [pokemon for pokemon in pokedex if int(pokemon.get("number")) >= start_at and int(pokemon.get("number")) <= end_at]

        if standardize:
            pokedex = [pokemon for pokemon in pokedex if pokemon.get("standard")]

        return pokedex


def get_pokedex(standardize=False, start_at=0, end_at=2000) -> list[Pokemon]:
    pokedex: list[PokemonData] = get_pokedex_data(standardize, start_at, end_at)
    pokemon_list: list[Pokemon] = []

    for pokemon in pokedex:
        pokemon_list.append(
            Pokemon(
                name=pokemon.get("name"),
                number=pokemon.get("number") if pokemon.get("standard") else pokemon.get("id"),
                primary_type=pokemon.get("primary_type"),
                secondary_type=pokemon.get("secondary_type"),
                generation=pokemon.get("generation"),
                region=pokemon.get("region", None),
                mega=pokemon.get("mega", False),
                form=pokemon.get("form", False),
                alternate_count=pokemon.get("alternate_count", 0),
                variants=pokemon.get("variants", []),
            )
        )

    return pokemon_list


# Organized based on the internal type ID used in Generation IX
def get_types() -> list[str]:
    return ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark", "fairy"]
