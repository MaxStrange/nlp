"""
This is the UI for the program. Call this script with:

python3 betrayal.py msg_pairs_one.yml msg_pairs_two.yml msg_pairs_n.yml

"""
import sys
import yaml


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need at least one YAML file.")
        print("USAGE:", "python3", sys.argv[0], "msg_pairs_one.yml msg_pairs_two.yml ... msg_pairs_n.yml")
        exit(1)
    elif len(sys.argv) < 5:
        print("WARNING: With less than three YAML files, you will be running in debug mode. You need three YAML files to make this work.")

    relationship_as_yaml = []
    for yam in sys.argv[1:]:
        print("Processing YAML file:", yam)
        with open(yam) as f:
            y = yaml.load(f)
            relationship_as_yaml.append(y)

    print(relationship_as_yaml)
