import csv
import json
import os


class Pokemon:
    def __init__(self, name, number, primary_type, secondary_type, generation, region, mega, alternate_forms):
        """Initializes a new instance of the Pokemon class."""
        self.name: str = name
        self.number: str = number
        self.primary_type: str = primary_type
        self.secondary_type: str = secondary_type
        self.generation: str = generation
        self.region: bool = region
        self.mega: bool = mega
        self.alternate_forms: list[str] = alternate_forms

    def __str__(self):
        """Returns a string representation of the Pokemon."""
        types = f"{self.primary_type}"
        if self.secondary_type:
            types += f"/{self.secondary_type}"
        return f"{self.name} (#{self.number}): {types}"

    def to_dict(self):
        """Converts the Pokemon instance into a dictionary suitable for JSON serialization."""
        obj = {
            "name": self.name,
            "number": self.number.split("_")[0],
            # ID is number-0 for non-alternate forms
            "id": (self.number.replace("_", "-") if "_" in self.number else f"{self.number}-0"),
            "primary_type": self.primary_type,
            "secondary_type": self.secondary_type if self.secondary_type else None,
            "generation": self.generation,
        }

        if self.region:
            obj["region"] = self.region

        if self.mega:
            obj["mega"] = self.mega

        if (len(self.alternate_forms)) > 0:
            obj["alternate_forms"] = (self.alternate_forms,)

        return obj


def process_data(pokedex_csv):
    pokemon_list: dict[str, Pokemon] = {}

    with open(pokedex_csv, encoding="utf-8-sig", newline="", mode="r") as file:
        pokedex = csv.DictReader(file)
        # print(pokedex.fieldnames)
        for entry in pokedex:
            pokdex_num = str(entry["No"])

            # Pokemon already in dict (alternate form)
            if pokemon_list.get(pokdex_num):
                pokemon = Pokemon(
                    name=entry["Name"],
                    number=entry["Branch_Code"],
                    primary_type=entry["Type1"],
                    secondary_type=entry["Type2"],
                    generation=entry["Generation"],
                    region=False,
                    mega=False,
                    alternate_forms=[],
                )

                if entry["Mega_Evolution_Flag"]:
                    pokemon.mega = True
                    pokemon_list[entry["Branch_Code"]] = pokemon
                elif entry["Region_Form"]:
                    pokemon.region = entry["Region_Form"]
                    pokemon_list[entry["Branch_Code"]] = pokemon
                else:
                    pokemon_list[pokdex_num].alternate_forms.append(entry["Name"])
                    """if pokemon_list[pokdex_num].primary_type == entry["Type1"]:
                        # Primary types match
                        #if pokemon_list[pokdex_num].secondary_type == entry["Type2"]:
                            # Primary and secondary types match

                    else:
                        print(f'{pokemon_list[pokdex_num].name} [{pokemon_list[pokdex_num].primary_type}, {pokemon_list[pokdex_num].secondary_type}] has alternate form {entry["Name"]} [{entry["Type1"]}, {entry["Type2"]}]')
                    """

            # New Pokemon
            else:
                pokemon = Pokemon(
                    name=entry["Original_Name"],
                    number=pokdex_num,
                    primary_type=entry["Type1"],
                    secondary_type=entry["Type2"],
                    generation=entry["Generation"],
                    region=False,
                    mega=False,
                    alternate_forms=[entry["Name"]] if entry["Original_Name"] != entry["Name"] else [],
                )

                pokemon_list[pokdex_num] = pokemon

    return pokemon_list


def main() -> None:
    dir = os.path.dirname(p=os.path.realpath(filename=__file__))
    # Sourced from https://www.kaggle.com/datasets/takamasakato/pokemon-all-status-data/data
    input_path = "data/Pokedex_Ver_SV2a.csv"
    additional_data_path = "data/pokedex_data_1011-1025.json"
    output_path = "pokemon_list.json"

    pokedex_data = process_data(pokedex_csv=os.path.join(dir, input_path))

    with open(os.path.join(dir, additional_data_path), mode="r") as additional_data_file:
        additional_data = json.load(additional_data_file)

    full_pokedex = [pokemon.to_dict() for pokemon in pokedex_data.values()] + additional_data

    with open(file=os.path.join(dir, output_path), mode="w") as file:
        json.dump(full_pokedex, file, indent=4)

    print(f"Data successfully written to {output_path}")


if __name__ == "__main__":
    main()
