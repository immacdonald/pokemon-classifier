import os

from PIL import Image

from pokedex.pokedex import get_pokedex

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def filter_pokemon(pokemon_name, pokemon_number):
    path = f"data/bulbapedia/{pokemon_number}_{pokemon_name.lower()}"
    directory = os.path.join(dir, path)

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_count = len(image_files)

    # Define thresholds
    minimum_size = 80
    transparency_threshold = 0.75
    transparency_minimum = 0.01
    deleted_count = 0

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

            # print(image_path.split('/')[-1], img.mode)

            # Remove palette or greyscale images
            if img.mode == "P" or img.mode == "L":
                os.remove(image_path)
                deleted_count += 1
                continue

            # Check for small images (less than 100px)
            if width < minimum_size or height < minimum_size:
                os.remove(image_path)
                deleted_count += 1
                continue

            # Remove non-transparent images
            if img.mode == "RGB":
                os.remove(image_path)
                deleted_count += 1
                continue

            # Check for transparent pixels if the image has an alpha channel
            if img.mode == "RGBA":
                transparent_pixels = 0
                total_pixels = width * height
                pixels = img.getdata()

                for pixel in pixels:
                    if pixel[3] == 0:  # Alpha channel at 0 means fully transparent
                        transparent_pixels += 1

                transparency_ratio = transparent_pixels / total_pixels
                if transparency_ratio > transparency_threshold or transparency_ratio < transparency_minimum:
                    os.remove(image_path)
                    deleted_count += 1

    print(f"{pokemon_name} filtered: started at {image_count} ended at {image_count - deleted_count} (deleted {deleted_count})")


def main() -> None:
    start_at = 0
    end_at = 1026
    pokemon_data = get_pokedex()

    for pokemon in pokemon_data:
        pokemon_number: str = pokemon.get("number")

        if int(pokemon_number) >= start_at and int(pokemon_number) <= end_at:
            if pokemon.get("mega") or pokemon.get("region"):
                print("Skipping")
            else:
                pokemon_name: str = pokemon.get("name")
                filter_pokemon(pokemon_name, pokemon_number)


if __name__ == "__main__":
    main()
