import os

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


# Creates directories relative to the file path
def create_directory(directory: str, base=False) -> str:
    directory_path: str = os.path.join(dir, directory) if not base else directory

    # Create folder to save images if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


# Initializes a Chrome driver that can be reused
def initialize_driver() -> WebDriver:
    # Setup Chrome options to run in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_service = Service(os.path.join(dir, "chromedriver"))

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    return driver


def scrape_images(url: str, output_directory: str, image_name: str, element_name: str, element_by_id: bool = True, driver=None) -> None:
    if driver is None:
        driver: WebDriver = initialize_driver()

    try:
        # Open the page
        driver.get(url=url)

        element_by = By.ID if element_by_id else By.CLASS_NAME

        # Wait for the desired div to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((element_by, element_name)))

        # Find the div and all image elements within it
        content_div = driver.find_element(element_by, element_name)
        images = content_div.find_elements(By.TAG_NAME, "img")

        for idx, img in enumerate(images):
            img_url = img.get_attribute("src")

            if img_url:
                # Fetch the image content
                img_data = requests.get(img_url).content
                # Define the file path to save the image
                img_path = os.path.join(output_directory, f"{image_name}_{idx+1}.jpg")

                # Save the image to the specified folder
                with open(img_path, "wb") as file:
                    file.write(img_data)

    finally:
        # Close the WebDriver
        driver.quit()
