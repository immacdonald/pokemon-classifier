import os

from PIL import Image

from shared.pokedex import get_pokedex

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def crop_pokemon(pokemon_name, pokemon_number):
    path = f"data/pokemon_cards/{pokemon_number}_{pokemon_name.lower()}"
    directory = os.path.join(dir, path)

    # Count images in directory
    image_files = [file for file in os.listdir(directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_count = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

            border_width = width / 9.5
            top_border_height = height / 9
            frame_height = height / 2.8
            bottom_crop = frame_height + top_border_height
            cropped_img = img.crop((border_width, top_border_height, width - border_width, bottom_crop))

            cropped_img.save(image_path)

    print(f"{pokemon_name} cropped {image_count} images")


def main() -> None:
    start_at = 896
    end_at = 1026
    pokemon_data = get_pokedex()

    for pokemon in pokemon_data:
        pokemon_number: str = pokemon.get("number")

        if int(pokemon_number) >= start_at and int(pokemon_number) <= end_at:
            if pokemon.get("mega") or pokemon.get("region"):
                print("Skipping")
            else:
                pokemon_name: str = pokemon.get("name")
                crop_pokemon(pokemon_name, pokemon_number)


if __name__ == "__main__":
    main()
