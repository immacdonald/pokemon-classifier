import os
import shutil

from pokedex.pokedex import get_pokedex, get_types
from pokedex.pokemon import Pokemon
from utility import get_pokemon_directory, sanitize_name

dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))


def add_to_dataset(basepath: str, directories: dict[str, str], pokemon: Pokemon, start_index: int = 0) -> int:
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

    if len(image_files) == 0:
        return start_index

    # Copy each image to the target directory with the new naming format
    for index, file in enumerate(image_files, start=1):
        new_filename = f"{pokemon_name}-{index + start_index}{os.path.splitext(file)[1]}"
        source_path = os.path.join(image_directory, file)
        destination_path = os.path.join(target_directory, new_filename)

        # Copy the image to the target directory with the new filename
        shutil.copy(source_path, destination_path)
        # print(f"Copied {file} to {destination_path}")

    print(f"All images for {pokemon_name} have been copied to {pokemon_type} dataset")

    return index + start_index


def generate_encoding(pokemon: Pokemon):
    types = get_types()
    encoding = ""
    for index in range(len(types)):
        if types[index] == pokemon.primary_type.lower() or (pokemon.secondary_type and types[index] == pokemon.secondary_type.lower()):
            encoding += "1"
        else:
            encoding += "0"

    return encoding


def add_to_multi_dataset(basepath: str, directory: str, pokemon: Pokemon, start_index: int = 0) -> None:
    pokemon_name = sanitize_name(pokemon.name)

    image_directory = os.path.join(basepath, get_pokemon_directory(pokemon))
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    encoding = generate_encoding(pokemon)

    if len(image_files) == 0:
        return start_index

    # Copy each image to the target directory with the new naming format
    for index, file in enumerate(iterable=image_files, start=1):
        new_filename = f"{pokemon_name}-{encoding}-{index + start_index}{os.path.splitext(file)[1]}"
        source_path = os.path.join(image_directory, file)
        destination_path = os.path.join(directory, new_filename)

        # Copy the image to the target directory with the new filename
        shutil.copy(source_path, destination_path)
        # print(f"Copied {file} to {destination_path}")

    print(f"All images for {pokemon_name} have been copied to the dataset")

    return index + start_index


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

    use_images = True
    use_cards = True
    one_hot_encoding = True

    basepath = os.path.join(dir, "data")

    if one_hot_encoding:
        directory: str = os.path.join(dir, "multidataset")

        # Only create if it does not already exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        for pokemon in pokemon_data:
            start_index = 1
            if use_images:
                start_index = add_to_multi_dataset(os.path.join(basepath, "bulbapedia"), directory, pokemon, start_index)
            if use_cards:
                add_to_multi_dataset(os.path.join(basepath, "pokemon_cards"), directory, pokemon, start_index)

    else:
        directories = create_type_directories()
        for pokemon in pokemon_data:
            start_index = 1
            if use_images:
                start_index = add_to_dataset(os.path.join(basepath, "bulbapedia"), directories, pokemon, start_index)
            if use_cards:
                add_to_dataset(os.path.join(basepath, "pokemon_cards"), directories, pokemon, start_index)


if __name__ == "__main__":
    main()
