import csv
import json
import os

from pokedex.pokemon import PokemonData

from .pokemon import Pokemon


def compile_data(pokedex_csv) -> dict[str, Pokemon]:
    pokemon_list: dict[str, Pokemon] = {}

    with open(pokedex_csv, encoding="utf-8-sig", newline="", mode="r") as file:
        pokedex = csv.DictReader(file)
        # print(pokedex.fieldnames)
        for entry in pokedex:
            pokedex_number = str(entry["No"])
            branch_code = str(entry["Branch_Code"].replace("_", "-"))

            # Pokemon already in dictionary (alternate form)
            if pokemon_list.get(pokedex_number):
                pokemon = Pokemon(
                    name=entry["Name"],
                    number=branch_code,
                    primary_type=entry["Type1"],
                    secondary_type=entry["Type2"],
                    generation=entry["Generation"],
                )

                if entry["Mega_Evolution_Flag"]:
                    pokemon.mega = True
                    pokemon_list[branch_code] = pokemon
                    pokemon_list[pokedex_number].alternate_count += 1
                elif entry["Region_Form"]:
                    pokemon.region = entry["Region_Form"]
                    pokemon_list[branch_code] = pokemon
                    pokemon_list[pokedex_number].alternate_count += 1
                else:
                    if pokemon_list[pokedex_number].primary_type == entry["Type1"]:
                        # Primary types match
                        if pokemon_list[pokedex_number].secondary_type == entry["Type2"]:
                            # Primary and secondary types match
                            pokemon_list[pokedex_number].variants.append(entry["Name"])
                            # print(f'Primary and secondary matches for {pokemon_list[pokedex_number].name} [{pokemon_list[pokedex_number].primary_type}, {pokemon_list[pokedex_number].secondary_type}] and {entry["Name"]} [{entry["Type1"]}, {entry["Type2"]}]')
                        else:
                            # Only primary types match so Pokemon is considered seperate
                            pokemon.form = True
                            pokemon_list[branch_code] = pokemon
                            pokemon_list[pokedex_number].alternate_count += 1
                            # print(f'Primary but not secondary matches for {pokemon_list[pokedex_number].name} [{pokemon_list[pokedex_number].primary_type}, {pokemon_list[pokedex_number].secondary_type}] and {entry["Name"]} [{entry["Type1"]}, {entry["Type2"]}]')

                    else:
                        # Different types so Pokemon is considered seperate
                        pokemon.form = True
                        pokemon_list[branch_code] = pokemon
                        pokemon_list[pokedex_number].alternate_count += 1

                        # print(f'{pokemon_list[pokedex_number].name} [{pokemon_list[pokedex_number].primary_type}, {pokemon_list[pokedex_number].secondary_type}] has alternate form with new types {entry["Name"]} [{entry["Type1"]}, {entry["Type2"]}]')

            # New Pokemon
            else:
                pokemon = Pokemon(
                    name=entry["Original_Name"],
                    number=pokedex_number,
                    primary_type=entry["Type1"],
                    secondary_type=entry["Type2"],
                    generation=entry["Generation"],
                    variants=[entry["Name"]] if entry["Original_Name"] != entry["Name"] else [],
                )

                pokemon_list[pokedex_number] = pokemon

    return pokemon_list


def main() -> None:
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))

    # Sourced from https://www.kaggle.com/datasets/takamasakato/pokemon-all-status-data/data
    input_path = "Pokedex_Ver_SV2.csv"
    output_path = "updated_pokemon_list.json"

    pokedex_data: dict[str, Pokemon] = compile_data(pokedex_csv=os.path.join(dir, input_path))

    full_pokedex: list[PokemonData] = [pokemon.to_dict() for pokemon in pokedex_data.values()]

    with open(file=os.path.join(dir, output_path), mode="w") as file:
        json.dump(full_pokedex, file, indent=4)

    print(f"Data successfully written to {output_path}")


if __name__ == "__main__":
    main()
