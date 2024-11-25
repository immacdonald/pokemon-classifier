import os

from PIL import Image

from pokedex.pokedex import get_pokedex
from pokedex.pokemon import Pokemon
from utility import get_pokemon_directory, safe_create_directory

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def filter_pokemon(pokemon: Pokemon):
    path = f"data/bulbapedia/{get_pokemon_directory(pokemon)}"
    directory = os.path.join(dir, path)

    exclusion_terms = ["funko", "komaplaza", "art_life", "binder", "box", "volume", "card", "coin", "print", "block", "paper", "quest", "trophy", "shirt", "sleeve", "deck"]

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_count = len(image_files)

    dry_run = False

    # Define thresholds
    minimum_width = 80
    minimum_height = 80
    transparency_threshold = 0.8
    transparency_minimum = 0.00
    deleted_count = 0

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

            # print(image_path.split('/')[-1], img.mode)

            # Check for small images (less than 100px)
            if width < minimum_width or height < minimum_height:
                if not dry_run:
                    os.remove(image_path)
                deleted_count += 1
                print("Deleting for size")
                continue

            # Remove palette or greyscale images
            if img.mode == "P" or img.mode == "L" or img.mode == "LA":
                if not dry_run:
                    os.remove(image_path)
                deleted_count += 1
                print("Deleting for format")
                continue

            # Remove images with titles that suggest the Pokemon is shiny
            filename = image_file.lower().split(".")[-2]
            if filename.endswith("_s") or "shiny" in filename:
                if not dry_run:
                    os.remove(image_path)
                deleted_count += 1
                continue

            # Check for transparent pixels if the image has an alpha channel
            transparent_pixels = 0
            total_pixels = img.width * img.height
            pixels = img.getdata()

            if img.mode == "RGBA":
                for pixel in pixels:
                    if pixel[3] == 0:
                        transparent_pixels += 1

                # Calculate ratios
                transparency_ratio = transparent_pixels / total_pixels

                # Check thresholds
                if transparency_ratio > transparency_threshold or transparency_ratio < transparency_minimum:
                    if not dry_run:
                        os.remove(image_path)
                    deleted_count += 1
                    continue
        
        # Check image name to remove unwanted names
        for exclusion in exclusion_terms:
            if exclusion in image_file.lower() and exclusion not in pokemon.name.lower():
                if not dry_run:
                    os.remove(image_path)
                deleted_count += 1
                print(f"Deleting for exclusion term {exclusion} for {image_file}")
                break

    if deleted_count > 0:
        print(f"{'Dry Run: ' if dry_run else ''} {pokemon.name} filtered: started at {image_count} ended at {image_count - deleted_count} (deleted {deleted_count})")


def filter_form(pokemon: Pokemon, base_form: Pokemon):
    path = f"data/bulbapedia/{get_pokemon_directory(base_form)}"
    directory = os.path.join(dir, path)

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    form_directory = safe_create_directory(os.path.join(dir, f"data/bulbapedia/{get_pokemon_directory(pokemon=pokemon)}"))

    moved = 0

    def move(image_path):
        nonlocal moved
        moved += 1
        os.rename(image_path, os.path.join(form_directory, image_file))

    for image_file in image_files:
        filename = image_file.split(".")[-2].lower()
        image_path = os.path.join(directory, image_file)
        if pokemon.mega:
            if "mega" in filename:
                move(image_path)
        elif pokemon.region:
            if pokemon.region == "Galarian":
                if "galar" in filename or filename.endswith("g"):
                    move(image_path)
            if pokemon.region == "Alolan":
                if "alola" in filename or filename.endswith("a"):
                    move(image_path)
            if pokemon.region == "Hisuian":
                if "hisui" in filename or filename.endswith("h"):
                    move(image_path)
            if pokemon.region == "Paldean":
                if "paldea" in filename or filename.endswith("p"):
                    move(image_path)

    print(f"Moved {moved} images to form directory {form_directory}")


def main() -> None:
    start_at = 0
    end_at = 1030
    pokemon_data: list[Pokemon] = get_pokedex(False, start_at, end_at)

    # Set when sorting has already been done and should not be repeated
    has_been_sorted_into_forms = True

    base_form: Pokemon | None = None

    for pokemon in pokemon_data:
        # print(get_pokemon_directory(pokemon))
        if has_been_sorted_into_forms:
            filter_pokemon(pokemon)
        else:
            if pokemon.standard:
                filter_pokemon(pokemon)

                if base_form:
                    base_form = None
                if pokemon.alternate_count > 0:
                    base_form = pokemon
            else:
                filter_form(pokemon, base_form)


if __name__ == "__main__":
    main()
