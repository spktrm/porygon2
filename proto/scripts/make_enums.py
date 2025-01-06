import json

with open("data/data/data.json") as f:
    DATA = json.load(f)


def toid(string: str) -> str:
    return "".join(c for c in string if c.isalnum() or c == "_").lower()


def generate_enum(title: str, data: dict[str, int], use_toid: bool = True):
    data_lines = [
        f"\t{title.upper()}_{((toid if use_toid else lambda x:x)(key)).upper()} = {idx};"
        for idx, (key, value) in enumerate(data.items())
    ]
    assert len(set(data.keys())) == len(set(data.values())), title
    return f"enum {title.capitalize()}Enum {{\n" + "\n".join(data_lines) + "\n}"


def main():
    enum_data = """syntax = "proto3";

package enums;

// This code is autogenerated by this script 
// proto/scripts/make_enums.py

"""

    for key, value in DATA.items():
        enum_data += generate_enum(key, value)
        enum_data += "\n\n"

    with open("proto/enums.proto", "w") as f:
        f.write(enum_data)


if __name__ == "__main__":
    main()
