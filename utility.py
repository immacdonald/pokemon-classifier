import os
import re

from pokedex.pokemon import Pokemon


def sanitize_name(name: str) -> str:
    name = name.lower()
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name


def get_pokemon_directory(pokemon: Pokemon):
    if pokemon.standard:
        return f"{pokemon.number.zfill(4)}_{sanitize_name(pokemon.name)}"
    else:
        return f"{pokemon.number.zfill(6)}_{sanitize_name(pokemon.name)}"
    

# Creates a directory if it does not already exist
def safe_create_directory(directory: str) -> str:
    # Create folder to save images if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory