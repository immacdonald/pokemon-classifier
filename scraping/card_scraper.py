import os

import requests
from shared.pokedex import get_pokedex
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def scrape_pokemon_images(pokemon_name, save_folder):
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Construct URL with the Pokémon name
    url = f"https://pkmncards.com/?s={pokemon_name}"

    # Setup Chrome options to run in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_service = Service(os.path.join(dir, "chromedriver"))

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    try:
        # Open the page
        driver.get(url=url)

        # Wait for the "genesis-content" div to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "genesis-content")))

        # Find the "genesis-content" div and all image elements within it
        content_div = driver.find_element(By.ID, "genesis-content")
        images = content_div.find_elements(By.TAG_NAME, "img")

        for idx, img in enumerate(images):
            img_url = img.get_attribute("src")

            if img_url:
                # Fetch the image content
                img_data = requests.get(img_url).content
                # Define the file path to save the image
                img_path = os.path.join(save_folder, f"{pokemon_name}_{idx+1}.jpg")

                # Save the image to the specified folder
                with open(img_path, "wb") as file:
                    file.write(img_data)

                print(f"Saved {img_path}")

    finally:
        # Close the WebDriver
        driver.quit()


def main():
    dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))
    save_folder: str = os.path.join(dir, "pokemon_cards")

    # Create folder to save images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    pokemon_data = get_pokedex()[2:4]
    for pokemon in pokemon_data:
        if pokemon.get("mega") or pokemon.get("region"):
            print("Skipping")
        else:
            pokemon_name: str = pokemon.get("name").lower()
            pokemon_number: str = pokemon.get("number")

            if pokemon_name:
                scrape_pokemon_images(pokemon_name, os.path.join(save_folder, f"{pokemon_number}_{pokemon_name}"))


if __name__ == "__main__":
    main()
