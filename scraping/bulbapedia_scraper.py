import os

from shared.pokedex import get_pokedex

from .scraper import create_directory, scrape_images


# Bulbapedia archives uses more precisely written names and case-sensitive URLs
def get_pokemon_category(pokemon_name: str, pokemon_number: str) -> str:
    if pokemon_number == "29":
        return "Nidoran♀"
    elif pokemon_number == "32":
        return "Nidoran♂"
    elif pokemon_number == "669":
        return "Flabébé"
    else:
        # Construct URL with the Pokémon name
        return pokemon_name.replace(" ", "_")


def scrape_pokemon_images(pokemon_name: str, pokemon_number: str, base_directory) -> None:
    save_folder = create_directory(os.path.join(base_directory, f"{pokemon_number}_{pokemon_name.lower()}"))

    url_base = "https://archives.bulbagarden.net/wiki/Category:"
    url_category = get_pokemon_category(pokemon_name, pokemon_number)
    url = f"{url_base}{url_category}"
    print(f"Scraping {url}")

    scrape_images(url, save_folder, pokemon_name.lower(), "mw-gallery-traditional", False)


def main() -> None:
    save_folder: str = create_directory("data/bulbapedia", True)
    print_prefix = "[Bulbapedia]"

    start_at = 0
    pokemon_data = get_pokedex()

    for pokemon in pokemon_data:
        pokemon_number: str = pokemon.get("number")

        if int(pokemon_number) >= start_at:
            if pokemon.get("mega") or pokemon.get("region"):
                print("Skipping")
            else:
                pokemon_name: str = pokemon.get("name")

                if pokemon_name:
                    print(f"{print_prefix} Started scraping {pokemon_name} #{pokemon_number}")
                    scrape_pokemon_images(pokemon_name, pokemon_number, save_folder)
                    print(f"{print_prefix} Finished scraping {pokemon_name}")


if __name__ == "__main__":
    main()
