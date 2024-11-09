from typing import NotRequired, TypedDict


class PokemonData(TypedDict):
    name: str
    number: str
    id: str
    primary_type: str
    secondary_type: str | None
    generation: str
    region: NotRequired[str]
    mega: NotRequired[bool]
    form: NotRequired[bool]
    is_alternate: NotRequired[bool]
    alternate_count: NotRequired[int]
    variants: NotRequired[list[str]]


def validate_pokemon(data: list[dict]) -> list[PokemonData]:
    validated_data: list[PokemonData] = []

    for pokemon in data:
        if "name" not in pokemon or not isinstance(pokemon["name"], str):
            raise ValueError("Invalid 'name' for unknown Pokemon")
        name: str = pokemon["name"]

        if "number" not in pokemon or not isinstance(pokemon["number"], str):
            raise ValueError(f"Invalid 'number' for {name}")

        if "id" not in pokemon or not isinstance(pokemon["id"], str):
            raise ValueError(f"Invalid 'id' for {name}")

        if "primary_type" not in pokemon or not isinstance(pokemon["primary_type"], str):
            raise ValueError(f"Invalid 'primary_type' for {name}")

        if "secondary_type" in pokemon and not (isinstance(pokemon["secondary_type"], str) or pokemon["secondary_type"] is None):
            raise ValueError(f"Invalid 'secondary_type' for {name}")

        if "generation" not in pokemon or not isinstance(pokemon["generation"], str):
            raise ValueError(f"Invalid 'generation' for {name}")

        # Optional fields
        if "region" in pokemon and not isinstance(pokemon["region"], str):
            raise ValueError(f"Invalid 'region' for {name}")

        if "mega" in pokemon and not isinstance(pokemon["mega"], bool):
            raise ValueError(f"Invalid 'mega' for {name}")

        if "form" in pokemon and not isinstance(pokemon["form"], bool):
            raise ValueError(f"Invalid 'form' for {name}")

        if "is_alternate" in pokemon and not isinstance(pokemon["is_alternate"], bool):
            raise ValueError(f"Invalid 'is_alternate' for {name}")

        if "alternate_count" in pokemon and not isinstance(pokemon["alternate_count"], int):
            raise ValueError(f"Invalid 'alternate_count' for {name}")

        if "variants" in pokemon and not isinstance(pokemon["variants"], list):
            raise ValueError(f"Invalid 'variants' for {name}")
        elif "variants" in pokemon:
            if not all(isinstance(variant, str) for variant in pokemon["variants"]):
                raise ValueError(f"All items in 'variants' must be strings for {name}")

        # Append validated data to the list
        validated_data.append(pokemon)

    return validated_data


class Pokemon:
    def __init__(self, name, number, primary_type, secondary_type, generation, region=None, mega=False, form=False, alternate_count=0, variants=[]) -> None:
        """Initializes a new instance of the Pokemon class."""
        self.name: str = name
        self.number: str = number
        self.primary_type: str = primary_type
        self.secondary_type: str | None = secondary_type
        self.generation: str = generation
        self.region: str | None = region
        self.mega: bool = mega
        self.form: bool = form
        self.alternate_count: int = alternate_count
        self.variants: list[str] = variants

    def __str__(self) -> str:
        """Returns a string representation of the Pokemon."""
        types = f"{self.primary_type}"
        if self.secondary_type:
            types += f"/{self.secondary_type}"
        return f"{self.name} (#{self.number}): {types}"

    def to_dict(self) -> PokemonData:
        """Converts the Pokemon instance into a dictionary suitable for JSON serialization."""
        data: PokemonData = {
            "name": self.name,
            "number": self.number.split("_")[0],
            # ID is number-0 for non-alternate forms
            "id": (self.number.replace("_", "-") if "_" in self.number else f"{self.number}-0"),
            "primary_type": self.primary_type,
            "secondary_type": self.secondary_type if self.secondary_type else None,
            "generation": self.generation,
        }

        if self.region:
            data["region"] = self.region
            data["is_alternate"] = True

        if self.mega:
            data["mega"] = self.mega
            data["is_alternate"] = True

        if self.form:
            data["form"] = self.form
            data["is_alternate"] = True

        if self.alternate_count > 0:
            data["alternate_count"] = self.alternate_count

        if (len(self.variants)) > 0:
            data["variants"] = self.variants

        return data
