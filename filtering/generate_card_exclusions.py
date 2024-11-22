import os

from scraping.scraper import scrape_and_save_images


def scrape_card_search(url, save_folder) -> None:
    scrape_and_save_images(url=url, output_directory=save_folder, image_name=None, element_name="genesis-content", element_by_id=True, next_text="next →")


def main() -> None:
    urls = [
        "https://pkmncards.com/?s=type%3Apokemon+rarity%3Arainbow-rare&sort=date&ord=auto&display=images",
        "https://pkmncards.com/?s=type%3Apokemon+rarity%3Arare-secret&sort=date&ord=auto&display=images",
        "https://pkmncards.com/?s=rarity%3Ahyper-rare&sort=date&ord=auto&display=images",
        "https://pkmncards.com/page/1/?s=rarity%3Ashiny-rare%2Crare-shiny-gx%2Cshiny-rare-v-or-vmax%2Crare-shining%2Cradiant-rare%2Cshiny-ultra-rare&sort=date&ord=auto&display=images",
        "https://pkmncards.com/?s=is%3Astar",
        "https://pkmncards.com/?s=type%3Apokemon+is%3Atag-team&sort=date&ord=auto&display=images",
        "https://pkmncards.com/?s=is%3Atera&sort=date&ord=auto&display=images",
    ]

    dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))
    save_folder: str = os.path.join(dir, "exclusions/pokemon_cards")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for url in urls:
        scrape_card_search(url, save_folder)


if __name__ == "__main__":
    main()
