import os
import shutil

from shared.pokedex import get_pokedex, get_types

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def add_to_dataset(basepath, directories, pokemon, file_suffix=""):
    pokemon_number = pokemon.get("number").lower()
    pokemon_name = pokemon.get("name").lower()
    pokemon_type = pokemon.get("primary_type").lower()
    pokemon_type_secondary = pokemon.get("secondary_type").lower() if pokemon.get("secondary_type") else None

    # Set primary type to "flying" if primary is "normal" and secondary is "flying"
    if pokemon_type == "normal" and pokemon_type_secondary == "flying":
        pokemon_type = pokemon_type_secondary

    image_directory = os.path.join(basepath, f"{pokemon_number}_{pokemon_name}")
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Get the target directory for the primary type
    target_directory = directories.get(pokemon_type)
    if not target_directory:
        print(f"Target directory for type '{pokemon_type}' not found.")
        return

    # Copy each image to the target directory with the new naming format
    for idx, file in enumerate(image_files, start=1):
        new_filename = f"{pokemon_name}{file_suffix}_{idx}{os.path.splitext(file)[1]}"
        source_path = os.path.join(image_directory, file)
        destination_path = os.path.join(target_directory, new_filename)

        # Copy the image to the target directory with the new filename
        shutil.copy(source_path, destination_path)
        # print(f"Copied {file} to {destination_path}")

    print(f"All images for {pokemon_name} have been copied to {pokemon_type} dataset")


# Create directories for each type
def create_type_directories():
    type_directories = {}

    for type in get_types():
        directory_path: str = os.path.join(dir, f"dataset/{type}")

        # Only create if it does not already exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        type_directories[type] = directory_path

    return type_directories


def main() -> None:
    start_at = 2
    end_at = 1026
    pokemon_data = get_pokedex()

    directories = create_type_directories()

    basepath = os.path.join(os.path.abspath(os.path.join(dir, "..")), "scraping/data")

    for pokemon in pokemon_data:
        pokemon_number: str = pokemon.get("number")

        if int(pokemon_number) >= start_at and int(pokemon_number) <= end_at:
            if not (pokemon.get("mega") or pokemon.get("region")):
                add_to_dataset(os.path.join(basepath, "bulbapedia"), directories, pokemon)
                add_to_dataset(os.path.join(basepath, "pokemon_cards"), directories, pokemon, file_suffix="_card")


if __name__ == "__main__":
    main()
