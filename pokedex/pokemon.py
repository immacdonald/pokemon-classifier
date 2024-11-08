class Pokemon:
    def __init__(self, name, number, primary_type, secondary_type, generation, region, mega, form, alternate_count, variants):
        """Initializes a new instance of the Pokemon class."""
        self.name: str = name
        self.number: str = number
        self.primary_type: str = primary_type
        self.secondary_type: str | None = secondary_type
        self.generation: str = generation
        self.region: str | None = region
        self.mega: bool | None = mega
        self.form: bool | None = form
        self.alternate_count: int | None = alternate_count
        self.variants: list[str] = variants

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
            obj["is_alternate"] = True

        if self.mega:
            obj["mega"] = self.mega
            obj["is_alternate"] = True

        if self.form:
            obj["form"] = self.form
            obj["is_alternate"] = True

        if self.alternate_count > 0:
            obj["alternate_count"] = self.alternate_count

        if (len(self.variants)) > 0:
            obj["variants"] = self.variants

        return obj