import os

from pokedex.pokedex import get_pokedex
from pokedex.pokemon import Pokemon

from .scraper import (
    create_directory,
    initialize_driver,
    sanitize_name,
    scrape_and_save_images,
)


def scrape_pokemon_images(pokemon: Pokemon, base_directory, driver) -> None:
    name = sanitize_name(pokemon.name)
    save_folder = create_directory(os.path.join(base_directory, f"{pokemon.number.zfill(4)}_{name}"))

    url = f"https://pkmncards.com/?s={pokemon.name.lower()}+type%3Apokemon&sort=date&ord=auto&display=images"

    scrape_and_save_images(url=url, output_directory=save_folder, image_name=None, element_name="genesis-content", element_by_id=True, next_text="next →", driver=driver)


def main() -> None:
    save_folder: str = create_directory("data/pokemon_cards", True)

    start_at = 484
    end_at = 1030
    pokemon_data: list[Pokemon] = get_pokedex(standardize=True, start_at=start_at, end_at=end_at)

    driver = initialize_driver()

    for pokemon in pokemon_data:
        print(f"Started scraping Cards for {pokemon.name} #{pokemon.number}")
        scrape_pokemon_images(pokemon, save_folder, driver)
        print(f"Finished scraping Cards for {pokemon.name}")

    # Close the driver upon a successful run of the scraper
    driver.close()


if __name__ == "__main__":
    main()
