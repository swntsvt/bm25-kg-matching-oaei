import os
import json
import re
from dotenv import load_dotenv
from rdflib import RDFS, URIRef

# Load the variables from the .env file into the environment
load_dotenv()


def get_config_from_env():
    """
    Reads all configuration variables from the loaded environment,
    parsing JSON strings back into Python dictionaries and lists.
    """

    # 1. Read simple string paths directly from the environment
    ontologies_base_path = os.getenv("ONTOLOGIES_BASE_PATH")
    references_base_path = os.getenv("REFERENCES_BASE_PATH")

    # Check if paths are defined (optional, but good practice)
    if not ontologies_base_path or not references_base_path:
        raise EnvironmentError("One or both base paths are missing from the .env file.")

    # 2. Read JSON strings and deserialize them

    # Load ontologies dictionary
    ontologies_json_str = os.getenv("ONTOLOGIES_JSON")
    if not ontologies_json_str:
        raise EnvironmentError("ONTOLOGIES_JSON is missing from the .env file.")
    ONTOLOGIES = json.loads(ontologies_json_str)

    # Load reference alignments list (will be a List of Lists)
    alignments_json_str = os.getenv("REFERENCE_ALIGNMENTS_JSON")
    if not alignments_json_str:
        raise EnvironmentError(
            "REFERENCE_ALIGNMENTS_JSON is missing from the .env file."
        )
    raw_alignments_list = json.loads(alignments_json_str)

    # Convert the list of lists back into a list of tuples for typical Python use
    REFERENCE_ALIGNMENTS = [tuple(item) for item in raw_alignments_list]

    # Return all loaded configuration data
    return {
        "ONTOLOGIES": ONTOLOGIES,
        "REFERENCE_ALIGNMENTS": REFERENCE_ALIGNMENTS,
        "ontologies_base_path": ontologies_base_path,
        "references_base_path": references_base_path,
    }


def get_kg_pair_paths():
    """
    Get the full file paths for each KG pair and their alignment
    based on the configuration loaded from the environment.

    ️Returns: Dict[str, Tuple[str, str, str]]
        A dictionary where each key is a concatenated string of source
        and target ontology names, and each value is a tuple containing
        the full paths to the source ontology, target ontology, and
        alignment file.
    """
    config = get_config_from_env()
    ontologies_dict = config["ONTOLOGIES"]
    ontologies_base_path = config["ontologies_base_path"]
    references_base_path = config["references_base_path"]

    kg_pair_paths = {}
    for alignment in config["REFERENCE_ALIGNMENTS"]:
        source_name, target_name, alignment_file = alignment
        source_path = os.path.join(ontologies_base_path, ontologies_dict[source_name])
        target_path = os.path.join(ontologies_base_path, ontologies_dict[target_name])
        alignment_path = os.path.join(references_base_path, alignment_file)
        kg_pair_paths[source_name + "-" + target_name] = (
            source_path,
            target_path,
            alignment_path,
        )

    return kg_pair_paths


def extract_label(g, uri_str):
    """Extract readable label from URI"""

    label = g.value(URIRef(uri_str), RDFS.label, None)
    if label:
        return str(label).strip()

    # Try to get the fragment or last part
    if "#" in uri_str:
        label = uri_str.split("#")[-1]
    elif "/" in uri_str:
        label = uri_str.split("/")[-1]
    else:
        label = uri_str

    # Clean up camelCase and snake_case
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    label = label.replace("_", " ").replace("-", " ")

    return label.strip()


def example_usage():
    """
    Example usage of the configuration loading and path construction functions.
    Demonstrates how to access various parts of the configuration and constructed paths.
    """
    # Example usage:
    try:
        config = get_config_from_env()

        print("✅ Configuration Loaded Successfully:")
        print(f"Ontologies Base Path: {config['ontologies_base_path']}")

        print(f"First Ontology Key: {list(config['ONTOLOGIES'].keys())[0]}")
        ontologies_dict = config["ONTOLOGIES"]
        print(
            f"First Ontology Value: {ontologies_dict[list(ontologies_dict.keys())[0]]}"
        )

        first_ontology_full_path = os.path.join(
            config["ontologies_base_path"],
            ontologies_dict[list(ontologies_dict.keys())[0]],
        )
        print(f"First Ontology Full Path: {first_ontology_full_path}")

        print(f"First Alignment Tuple: {config['REFERENCE_ALIGNMENTS'][0]}")

        first_alignment = config["REFERENCE_ALIGNMENTS"][0]
        first_source_name = first_alignment[0]
        first_source_path = os.path.join(
            f"{config['ontologies_base_path']}", f"{ontologies_dict[first_source_name]}"
        )
        first_target_name = first_alignment[1]
        first_target_path = os.path.join(
            f"{config['ontologies_base_path']}", f"{ontologies_dict[first_target_name]}"
        )
        first_alignment_path = os.path.join(
            f"{config['references_base_path']}", f"{first_alignment[2]}"
        )
        print(f"First Source Full Path: {first_source_path}")
        print(f"First Target Full Path: {first_target_path}")
        print(f"First Alignment Full Path: {first_alignment_path}")

        kg_pair_paths = get_kg_pair_paths()
        print("\n✅ KG Pair Paths Constructed Successfully:")
        print(f"Total KG Pairs: {len(kg_pair_paths)}")
        print(f"First KG Pair Key: {list(kg_pair_paths.keys())[0]}")
        print(f"First KG Pair Paths: {list(kg_pair_paths.values())[0]}")

    except EnvironmentError as e:
        print(f"❌ Configuration Error: {e}")


if __name__ == "__main__":
    example_usage()
