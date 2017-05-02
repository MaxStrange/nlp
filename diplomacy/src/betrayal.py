"""
This is the UI for the program. Call this script with:

python3 betrayal.py msg_pairs_one.yml msg_pairs_two.yml msg_pairs_n.yml

"""
import data
import inference
import sys
import yaml

def _convert_relationship_from_yaml(rel_as_yam):
    """
    Converts a list of YAML dicts into Relationship objects.
    """
    return inference.get_relationship(rel_as_yam)

def _predict(rel):
    """
    Predicts the betrayal probabilities and returns them as an inference.Output object.
    """
    # TODO
    return rel

def _load_yaml_files(files):
    """
    Loads the files found at the given file paths into YAML dictionaries
    and concatenates them into a single list of dicts and returns it.
    """
    relationship_as_yaml = []
    for yam in files:
        with open(yam) as f:
            y = yaml.load(f)
            relationship_as_yaml.append(y)
    return relationship_as_yaml

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need at least one YAML file.")
        print("USAGE:", "python3", sys.argv[0], "msg_pairs_one.yml msg_pairs_two.yml ... msg_pairs_n.yml")
        exit(1)
    elif len(sys.argv) < 5:
        print("WARNING: With less than three YAML files, you will be running in debug mode. You need three YAML files to make this work.")

    print("Loading YAML files...")
    relationship_as_yaml = _load_yaml_files(sys.argv[1:])
    print("Converting YAML files into a single relationship b/w the two players and doing NLP analysis...")
    relationship = _convert_relationship_from_yaml(relationship_as_yaml)
    print("Predicting the betrayal likelihoods...")
    betrayals = _predict(relationship)
    print(betrayals)

