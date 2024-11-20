import os
import re
from typing import TypedDict

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

class ScrapedImageData(TypedDict):
    name: str
    data: any

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


# Creates directories relative to the file path
def create_directory(directory: str, base=False) -> str:
    directory_path: str = os.path.join(dir, directory) if base else directory

    # Create folder to save images if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


def sanitize_name(name: str) -> str:
    name = name.lower()
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name


# Initializes a Chrome driver that can be reused
def initialize_driver() -> WebDriver:
    # Setup Chrome options to run in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_service = Service(os.path.join(dir, "chromedriver"))

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    return driver

def scrape_and_save_images(url: str, output_directory: str, image_name: str | None, element_name: str, element_by_id: bool = True, next_text: str | None = None, driver: WebDriver = None) -> None:
    images: list[ScrapedImageData] = scrape_images(url, element_name, element_by_id, next_text, driver)

    print(f"Successfully scraped {len(images)} images for {output_directory.split('/')[-1]}")
    for index, image in enumerate(images):
        # Define the file path to save the image
        img_path = os.path.join(output_directory, f"{image_name}_{index+1}.jpg" if image_name else image["name"])

        # Save the image to the specified folder
        with open(img_path, "wb") as file:
            file.write(image["data"])


def scrape_images(url: str, element_name: str, element_by_id: bool = True, next_text: str | None = None, driver: WebDriver = None) -> list[ScrapedImageData]:
    if driver is None:
        print("Creating webdriver")
        driver: WebDriver = initialize_driver()

    driver.get(url=url)
    element_by = By.ID if element_by_id else By.CLASS_NAME

    all_images: list[ScrapedImageData] = []

    # Scrape tracks the page status for paginated searches
    scrape = 1

    while scrape > 0:
        try:
            print(f"Scraping {url} page {scrape}")
            # Wait for the desired div to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((element_by, element_name)))

            # Find the div and all image elements within it
            content_div = driver.find_element(element_by, element_name)
            images = content_div.find_elements(By.TAG_NAME, "img")

            for img in images:
                img_url = img.get_attribute("src")

                if img_url:
                    img_name = img_url.split('/')[-1]
                    #print(f"Getting {img_name}")
                    # Fetch the image content
                    img_data = requests.get(img_url).content

                    all_images.append({
                        "name": img_name,
                        "data": img_data
                    })


            if next_text:
                next_page_link = driver.find_element(By.LINK_TEXT, next_text)
                if next_page_link:
                    scrape += 1
                    next_page_link.click()
                else:
                    scrape = 0
            else:
                scrape = 0

        except:
            # Close the WebDriver
            driver.quit()
    
    return all_images
