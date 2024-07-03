import json

with open("data/data/data.json") as f:
    DATA = json.load(f)


def toid(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def generate_enum(title: str, data: dict[str, int]):
    data = [f"\t{toid(key)} = {value + 1};" for key, value in data.items()]
    return f"enum {title} {{\n" + "\n".join(data) + "\n}"


def main():
    enum_data = 'syntax = "proto3";\n\n'

    for key, value in DATA.items():
        enum_data += generate_enum(key, value)
        enum_data += "\n\n"

    with open("proto/enums.proto", "w") as f:
        f.write(enum_data)


if __name__ == "__main__":
    main()
