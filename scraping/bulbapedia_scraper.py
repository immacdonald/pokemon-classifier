import os

from pokedex.pokedex import get_pokedex
from pokedex.pokemon import Pokemon

from .scraper import create_directory, initialize_driver, sanitize_name, scrape_and_save_images


# Bulbapedia archives uses more precisely written names and case-sensitive URLs
def get_bulbapedia_name(pokemon_name: str, pokemon_number: str) -> str:
    if pokemon_number == "29":
        return "Nidoran♀"
    elif pokemon_number == "32":
        return "Nidoran♂"
    elif pokemon_number == "669":
        return "Flabébé"
    else:
        # Construct URL with the Pokémon name
        return pokemon_name.replace(" ", "_")



def scrape_pokemon_images(pokemon: Pokemon, base_directory, driver) -> None:
    name = sanitize_name(pokemon.name)
    save_folder = create_directory(os.path.join(base_directory, f"{pokemon.number.zfill(4)}_{name}"))

    url_base = "https://archives.bulbagarden.net/wiki/Category:"
    url_search = get_bulbapedia_name(pokemon.name, pokemon.number)
    url = f"{url_base}{url_search}"

    scrape_and_save_images(url=url, output_directory=save_folder, image_name=None, element_name="mw-gallery-traditional", element_by_id=False, next_text='next page', driver=driver)


def main() -> None:
    save_folder: str = create_directory("new_data/bulbapedia", True)

    start_at = 745
    end_at = 1030
    pokemon_data: list[Pokemon] = get_pokedex(standardize=True, start_at=start_at, end_at=end_at)

    driver = initialize_driver()

    for pokemon in pokemon_data:
        print(f"Started scraping Bulbapedia {pokemon.name} #{pokemon.number}")
        scrape_pokemon_images(pokemon, save_folder, driver)
        print(f"Finished scraping Bulbapedia {pokemon.name}")

    # Close the driver upon a successful run of the scraper
    driver.close()


if __name__ == "__main__":
    main()
