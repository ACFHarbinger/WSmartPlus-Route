import os
import sys

import yaml


def to_bash_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, dict):
        # Convert dict to bash associative array: ( ["key1"]="val1" ["key2"]="val2" )
        # Note: Bash 4.0+ required.
        entries = []
        for k, v in value.items():
            entries.append(f'["{str(k)}"]="{str(v)}"')
        return f'({" ".join(entries)})'
    elif isinstance(value, list):
        # Convert list to bash array: ("item1" "item2")
        # Quote each item to handle spaces
        items = [f'"{str(v)}"' for v in value]
        return f'({" ".join(items)})'
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # String, quote it
        return f'"{str(value)}"'


def main():
    if len(sys.argv) != 2:
        print("Usage: python yaml_to_env.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}", file=sys.stderr)
            sys.exit(1)

    if not config:
        return

    for key, value in config.items():
        bash_var_name = key.upper()
        bash_value = to_bash_value(value)
        if isinstance(value, dict):
            # Associative arrays must be declared with -A
            print(f"declare -A {bash_var_name}={bash_value}")
        else:
            print(f"export {bash_var_name}={bash_value}")


if __name__ == "__main__":
    main()
