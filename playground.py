import os
from collections import defaultdict
from pprint import pprint

from pokedex.pokedex import get_pokedex

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


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


def count_images(pokemon_data):
    sprite_images = 0
    card_images = 0
    images = 0

    types = defaultdict(lambda: {"sprite_images": 0, "card_images": 0, "images": 0})

    for pokemon in pokemon_data:
        pokemon_number: str = pokemon.get("number")
        pokemon_name: str = pokemon.get("name")

        if not (pokemon.get("mega") or pokemon.get("region")):
            type = pokemon["primary_type"].lower()

            path = f"scraping/data/pokemon_cards/{pokemon_number}_{pokemon_name.lower()}"

            files = [file for file in os.listdir(os.path.join(dir, path)) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

            for _ in files:
                card_images += 1
                types[type]["card_images"] += 1
                types[type]["images"] += 1

            path = f"scraping/data/bulbapedia/{pokemon_number}_{pokemon_name.lower()}"

            files = [file for file in os.listdir(os.path.join(dir, path)) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

            for _ in files:
                sprite_images += 1
                types[type]["sprite_images"] += 1
                types[type]["images"] += 1

    images = sprite_images + card_images
    print(f"Dataset contains {sprite_images} sprites and {card_images} cards for {images} total")
    types = dict(types)
    pprint(types)


def main() -> None:
    pokedex = get_pokedex()
    print(len(pokedex), "Pokemon Total")
    # pprint(count_types(pokedex))
    count_images(pokedex)


if __name__ == "__main__":
    main()
