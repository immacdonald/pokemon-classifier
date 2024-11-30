import os
import shutil

from pokedex.pokedex import get_pokedex, get_types
from pokedex.pokemon import Pokemon
from utility import get_pokemon_directory, sanitize_name

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def add_to_dataset(basepath: str, directories: dict[str, str], pokemon: Pokemon, file_suffix: str = "") -> None:
    pokemon_name = sanitize_name(pokemon.name)
    pokemon_type = pokemon.primary_type.lower()
    pokemon_type_secondary = pokemon.secondary_type.lower() if pokemon.secondary_type else None

    # Set primary type to "flying" if primary is "normal" and secondary is "flying"
    if pokemon_type == "normal" and pokemon_type_secondary == "flying":
        pokemon_type = pokemon_type_secondary

    image_directory = os.path.join(basepath, get_pokemon_directory(pokemon))
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Get the target directory for the primary type
    target_directory = directories.get(pokemon_type)
    if not target_directory:
        print(f"Target directory for type '{pokemon_type}' not found.")
        return

    # Copy each image to the target directory with the new naming format
    for index, file in enumerate(image_files, start=1):
        new_filename = f"{pokemon_name}{file_suffix}_{index}{os.path.splitext(file)[1]}"
        source_path = os.path.join(image_directory, file)
        destination_path = os.path.join(target_directory, new_filename)

        # Copy the image to the target directory with the new filename
        shutil.copy(source_path, destination_path)
        # print(f"Copied {file} to {destination_path}")

    print(f"All images for {pokemon_name} have been copied to {pokemon_type} dataset")


# Create directories for each type
def create_type_directories():
    type_directories: dict[str, str] = {}

    for type in get_types():
        directory_path: str = os.path.join(dir, f"dataset/{type}")

        # Only create if it does not already exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        type_directories[type] = directory_path

    return type_directories


def main() -> None:
    pokemon_data = get_pokedex()

    directories = create_type_directories()

    use_images = True
    use_cards = True

    basepath = os.path.join(os.path.abspath(os.path.join(dir, "..")), "filtering/data")

    for pokemon in pokemon_data:
        if use_images:
            add_to_dataset(os.path.join(basepath, "bulbapedia"), directories, pokemon)
        if use_cards:
            add_to_dataset(os.path.join(basepath, "pokemon_cards"), directories, pokemon, file_suffix="_card")


if __name__ == "__main__":
    main()
