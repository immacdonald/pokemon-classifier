import os

from PIL import Image

from pokedex.pokedex import get_pokedex
from pokedex.pokemon import Pokemon
from utility import get_pokemon_directory, safe_create_directory

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def crop_cards(pokemon: Pokemon):
    path = f"data/pokemon_cards/{get_pokemon_directory(pokemon)}"
    directory = os.path.join(dir, path)

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_count = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

            border_left_width = width / 7.25
            border_right_width = width / 8.5
            top_border_height = height / 7.25
            frame_height = height / 3
            bottom_crop_height = frame_height + top_border_height
            cropped_img = img.crop((border_left_width, top_border_height, width - border_right_width, bottom_crop_height))

            cropped_img.save(image_path)

    print(f"{pokemon.name} cropped {image_count} images")


def filter_form(pokemon: Pokemon, base_form: Pokemon):
    path = f"data/pokemon_cards/{get_pokemon_directory(base_form)}"
    directory = os.path.join(dir, path)

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    form_directory = safe_create_directory(os.path.join(dir, f"data/pokemon_cards/{get_pokemon_directory(pokemon=pokemon)}"))

    moved = 0

    def move(image_path):
        nonlocal moved
        moved += 1
        os.rename(image_path, os.path.join(form_directory, image_file))

    for image_file in image_files:
        filename = image_file.split(".")[-2].lower()
        image_path = os.path.join(directory, image_file)
        if pokemon.mega:
            if "mega" in filename or "m_" in filename:
                move(image_path)
        elif pokemon.region:
            if pokemon.region == "Galarian":
                if "galar" in filename:
                    move(image_path)
            if pokemon.region == "Alolan":
                if "alola" in filename:
                    move(image_path)
            if pokemon.region == "Hisuian":
                if "hisui" in filename:
                    move(image_path)
            if pokemon.region == "Paldean":
                if "paldea" in filename:
                    move(image_path)

    print(f"Moved {moved} images to form directory {form_directory}")


def check_against_exclusions(pokemon: Pokemon):
    path = f"data/pokemon_cards/{get_pokemon_directory(pokemon)}"
    directory = os.path.join(dir, path)
    exclusion_directory = os.path.join(dir, "exclusions/pokemon_cards")

    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_count = len(image_files)

    exclusion_files = [file.lower() for file in os.listdir(exclusion_directory)]

    deleted_count = 0

    for file in image_files:
        if pokemon.number is "6":
            print(image_files)
        if file.lower() in exclusion_files:
            print(file)
            image_path = os.path.join(directory, file)
            os.remove(image_path)
            deleted_count += 1
            continue

    if deleted_count > 0:
        print(f"{pokemon.name} filtered: started at {image_count} ended at {image_count - deleted_count} (deleted {deleted_count})")


def main() -> None:
    start_at = 0
    end_at = 1030
    pokemon_data: list[Pokemon] = get_pokedex(False, start_at, end_at)

    # Set when sorting, cropping, and excluding has already been done and should not be repeated
    has_been_sorted_into_forms = True
    has_been_cropped = True
    has_been_checked_against_exclusions = False

    base_form: Pokemon | None = None

    for pokemon in pokemon_data:
        # print(get_pokemon_directory(pokemon))
        if has_been_sorted_into_forms:
            if not has_been_checked_against_exclusions:
                check_against_exclusions(pokemon)

            if not has_been_cropped:
                crop_cards(pokemon)
        else:
            if pokemon.standard:
                if not has_been_checked_against_exclusions:
                    check_against_exclusions(pokemon)

                if not has_been_cropped:
                    crop_cards(pokemon)

                if base_form:
                    base_form = None
                if pokemon.alternate_count > 0:
                    base_form = pokemon
            else:
                filter_form(pokemon, base_form)


if __name__ == "__main__":
    main()
