import itertools
import logging
import os
import time
import bm25s
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd

from tqdm import tqdm

from utils import get_kg_pair_paths, extract_label

# Hits@K values to evaluate
K_SET = (1, 5, 10, 20, 50)
K = 50

stop_words = set(stopwords.words("english"))
stop_words.add("type")
punctuation = set(string.punctuation)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Create reference alginments directory if it doesn't exist
references_dir = os.path.join(data_dir, "reference_alignments")
os.makedirs(references_dir, exist_ok=True)

# Create candidates directory if it doesn't exist
candidates_dir = os.path.join(data_dir, "candidates")
os.makedirs(candidates_dir, exist_ok=True)

# Create results directory if it doesn't exist
results_dir = os.path.join(data_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "bm25_candidate_generator.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def extract_entities(g):
    """
    Extract all entities and their types from an RDF file

    Args:
        g (rdflib.Graph): RDF graph to extract entities from

    Returns:
        set: Set of entity URIs and their types found in the RDF file
    """

    # Set to store unique entities and their types
    entities = set()

    # Query for all resources that are of type rdfs:Class
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?entity ?entity_type WHERE {
        ?entity ?p ?o .
        ?entity a ?type .
        VALUES ?type { owl:Class rdf:Property } .  # Limit to Classes and Properties
        BIND (
            IF (
                ?type = owl:Class ,
                "Class" ,
                IF (
                    ?type = rdf:Property ,
                    "Predicate" ,
                    "Instance"
                )
            )
            AS ?entity_type
        )
    }
    """

    try:
        results = g.query(query)
        for row in results:
            entities.add((str(row.entity), str(row.entity_type)))  # type: ignore
    except Exception as e:
        logger.error(f"Error executing query: {e}")

    logger.info(f"Total number of entities found through query: {len(entities)}")

    return entities


def get_entity_label(g, uri_str, labels):
    """Get or extract entity label with caching"""
    if labels.get(uri_str):
        return labels[uri_str]
    else:
        label = extract_label(g, uri_str)
        labels[uri_str] = label
        return label


def get_subject_verbalization(g, uri, labels):
    """Verbalize the subject entity given its URI.
    Currently, only handles URIRefs, skips blank nodes.
    """
    uri_str = str(uri)
    if isinstance(uri, URIRef):
        local = get_entity_label(g, uri_str, labels)
        return local
    elif isinstance(uri, BNode):
        logger.info(f"Skipping blank node for triple with URI {uri_str}")
        return None  # Skip blank nodes for verbalization


def get_object_verbalization(g, uri, labels):
    """Verbalize the object entity given its URI.
    Currently, only handles URIRefs and literals, skips blank nodes.
    """
    uri_str = str(uri)
    if isinstance(uri, URIRef):
        local = get_entity_label(g, uri_str, labels)
    elif isinstance(uri, BNode):
        logger.info(f"Skipping blank node object for triple with URI {uri_str}")
        return None  # Skip blank nodes for verbalization
    else:
        local = uri_str
    return local


def tokenize_entity_verbalization(g, uri_str):
    """
    Verbalize an entity given its URI by extracting all triples where it is the subject,
    predicate or object. Currently, only handles URIRefs and literals, skips blank nodes.

    Args:
        g (Graph): The RDF graph
        uri_str (str): The URI to verbalize
    Returns:
        list: The verbalized form of the URI, after removing stop words and punctuation,
        as a list of tokens
    """

    entity_paragraph = ""
    labels = {}

    # get all triples where the given uri is present as subject
    triples = g.predicate_objects(URIRef(uri_str), unique=True)
    e_local = get_entity_label(g, uri_str, labels)
    entity_paragraph += f"{e_local}"

    # verbalize where given uri is subject
    for p, o in triples:
        if p == RDFS.label:
            continue  # Skip rdfs:label triples
        if p == RDF.type and (o == RDFS.Class or o == OWL.Class):
            continue  # Skip rdf:type Class triples
        p_local = get_entity_label(g, str(p), labels)

        o_local = get_object_verbalization(g, o, labels)
        if o_local is None:
            continue  # Skip if object verbalization identified as blank node

        entity_paragraph += f"{p_local} {o_local}. "

    # verbalize where given uri is object
    for s, p in g.subject_predicates(URIRef(uri_str), unique=True):
        s_local = get_subject_verbalization(g, s, labels)
        if s_local is None:
            continue  # Skip if subject verbalization identified as blank node

        p_local = get_entity_label(g, str(p), labels)
        entity_paragraph += f"{s_local} {p_local} "

    # verbalize where given uri is predicate
    for s, o in g.subject_objects(URIRef(uri_str), unique=True):
        s_local = get_subject_verbalization(g, s, labels)
        if s_local is None:
            continue  # Skip if subject verbalization identified as blank node

        o_local = get_object_verbalization(g, o, labels)
        if o_local is None:
            continue  # Skip if object verbalization identified as blank node

        entity_paragraph += f"{s_local} {o_local}. "

    tokens = word_tokenize(entity_paragraph.strip().lower())
    filtered_tokens = [
        word for word in tokens if word not in stop_words and word not in punctuation
    ]

    return filtered_tokens


def entity_verbalization(g, uri_str):
    """
    Verbalize an entity given its URI by extracting all triples where it is the subject,
    predicate or object. Currently, only handles URIRefs and literals, skips blank nodes.

    Args:
        g (Graph): The RDF graph
        uri_str (str): The URI to verbalize
    Returns:
        str: The verbalized form of the URI, after removing stop words and punctuation as a paragraph
    """

    entity_paragraph = ""
    labels = {}

    # get all triples where the given uri is present as subject
    triples = g.predicate_objects(URIRef(uri_str), unique=True)
    e_local = get_entity_label(g, uri_str, labels)
    entity_paragraph += f"{e_local}"

    # verbalize where given uri is subject
    for p, o in triples:
        if p == RDFS.label:
            continue  # Skip rdfs:label triples
        if p == RDF.type and (o == RDFS.Class or o == OWL.Class):
            continue  # Skip rdf:type Class triples
        p_local = get_entity_label(g, str(p), labels)

        o_local = get_object_verbalization(g, o, labels)
        if o_local is None:
            continue  # Skip if object verbalization identified as blank node

        entity_paragraph += f"{p_local} {o_local}. "

    # verbalize where given uri is object
    for s, p in g.subject_predicates(URIRef(uri_str), unique=True):
        s_local = get_subject_verbalization(g, s, labels)
        if s_local is None:
            continue  # Skip if subject verbalization identified as blank node

        p_local = get_entity_label(g, str(p), labels)
        entity_paragraph += f"{s_local} {p_local} "

    # verbalize where given uri is predicate
    for s, o in g.subject_objects(URIRef(uri_str), unique=True):
        s_local = get_subject_verbalization(g, s, labels)
        if s_local is None:
            continue  # Skip if subject verbalization identified as blank node

        o_local = get_object_verbalization(g, o, labels)
        if o_local is None:
            continue  # Skip if object verbalization identified as blank node

        entity_paragraph += f"{s_local} {o_local}. "

    return entity_paragraph.strip().lower()


def generate_entity_paragraphs(file_path):
    """
    Generate verbalized paragraphs for each entity in the RDF file.

    Args:
        file_path (str): Path to the RDF file

    Returns:
        tuple: Two dictionaries containing verbalized paragraphs for
               classes and predicates respectively.
    """
    g = Graph()

    try:
        g.parse(file_path, format="xml")
    except Exception as e:
        logger.error(f"Error parsing RDF file: {e}")
        return {}

    entities = extract_entities(g)

    class_corpus = {}
    predicate_corpus = {}

    for entity, entity_type in entities:
        if entity_type == "Class":
            class_corpus[entity] = entity_verbalization(g, entity)
        elif entity_type == "Predicate":
            predicate_corpus[entity] = entity_verbalization(g, entity)
        else:
            logger.warning(f"Entity {entity} of type {entity_type} skipped.")

    logger.info(f"Total number of Class entities: {len(class_corpus)}")
    logger.info(f"Total number of Predicate entities: {len(predicate_corpus)}")
    return class_corpus, predicate_corpus


def load_reference_alignments(source_file, target_file, alignment_file):
    """Load the reference alignment file"""

    g = Graph()
    try:
        g.parse(alignment_file, format="xml")
        g.parse(source_file, format="xml")
        g.parse(target_file, format="xml")

    except Exception as e:
        logger.error(f"Error parsing alignment RDF file: {e}")
        return list()

    # Find all classes (resources that are rdf:type rdfs:Class)
    class_alignments = list()
    predicate_alignments = list()
    instance_alignments = list()

    # Query for all resources that are of type rdfs:Class
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX aln: <http://knowledgeweb.semanticweb.org/heterogeneity/alignment>
    SELECT DISTINCT ?source ?target ?entity_type ?source_type ?target_type WHERE {
        ?align a aln:Cell
            ; aln:entity1 ?source
            ; aln:entity2 ?target .
        ?source a ?source_type .
        ?target a ?target_type .
        # VALUES ?source_type { owl:Class rdf:Property } .  # Limit to Classes and Properties
        BIND (
            IF (
                ?source_type = owl:Class ,
                "Class" ,
                IF (
                    ?source_type = rdf:Property ,
                    "Predicate" ,
                    "Instance"
                )
            )
            AS ?entity_type
        )
    } ORDER BY ?entity_type
    """

    try:
        results = g.query(query)
        for row in results:
            source = str(row.source)  # type: ignore
            target = str(row.target)  # type: ignore
            entity_type = str(row.entity_type)  # type: ignore
            if entity_type == "Class":
                class_alignments.append((source, target))
            elif entity_type == "Predicate":
                predicate_alignments.append((source, target))
            elif entity_type == "Instance":
                source_class = str(row.source_type)  # type: ignore
                target_class = str(row.target_type)  # type: ignore
                instance_alignments.append((source, target, source_class, target_class))
            else:
                logger.warning(
                    f"Alignment between {source} and {target} of type {entity_type} skipped."
                )
    except Exception as e:
        logger.error(f"Error executing query: {e}")

    logger.info(
        f"Total number of class alignments found through query: {len(class_alignments)}"
    )
    logger.info(
        f"Total number of predicate alignments found through query: {len(predicate_alignments)}"
    )
    logger.info(
        f"Total number of instance alignments found through query: {len(instance_alignments)}"
    )

    # save reference alignments to csv files
    kg_pair = os.path.basename(alignment_file).replace(".rdf", "")
    class_output_file = os.path.join(references_dir, f"class_reference_{kg_pair}.csv")
    predicate_output_file = os.path.join(
        references_dir, f"predicate_reference_{kg_pair}.csv"
    )
    instance_output_file = os.path.join(
        references_dir, f"instance_reference_{kg_pair}.csv"
    )
    save_matches_to_csv(class_alignments, class_output_file)
    save_matches_to_csv(predicate_alignments, predicate_output_file)
    save_instance_matches_to_csv(instance_alignments, instance_output_file)

    return class_alignments, predicate_alignments, instance_alignments


def get_candidate_matches(source_paragraphs, target_paragraphs, k1, b):
    """
    Get candidate matches using BM25

    Args:
        source_paragraphs (dict): Dictionary of source entity URIs to their verbalized paragraphs
        target_paragraphs (dict): Dictionary of target entity URIs to their verbalized paragraphs
        k1 (float): BM25 k1 parameter
        b (float): BM25 b parameter

    Returns:
        list: List of candidate matches as tuples of (source_entity_uri, target_entity_uri)
    """
    target_ids = list(target_paragraphs.keys())
    target_corpus = list(target_paragraphs.values())

    target_corpus_tokens = bm25s.tokenize(target_corpus)
    bm25 = bm25s.BM25(k1=k1, b=b)
    bm25.index(target_corpus_tokens)

    candidate_matches = []

    for source_uri_str, source_paragraph in source_paragraphs.items():
        # Tokenize the query
        query_tokens = bm25s.tokenize(source_paragraph)
        target_indices, scores = bm25.retrieve(
            query_tokens,
            k=K,
            show_progress=True,
            n_threads=-1,
        )

        for i in range(len(target_indices[0])):
            target_idx = target_indices[0][i]
            # score = scores[0][i]  # FIXME: currently not used - may be useful for future analysis
            target_uri_str = target_ids[target_idx]
            candidate_matches.append((source_uri_str, target_uri_str))

    return candidate_matches


def save_matches_to_csv(matches, output_file):
    # save candidate matches to a csv file
    try:
        with open(output_file, "w") as f:
            f.write("source,target\n")
            for src, tgt in matches:
                f.write(f"{src},{tgt}\n")
    except Exception as e:
        logger.error(f"Error saving matches to csv: {e}")


def save_instance_matches_to_csv(matches, output_file):
    # save candidate matches to a csv file
    try:
        with open(output_file, "w") as f:
            f.write("source_instance,target_instance,source_class,target_class\n")
            for src, tgt, src_class, tgt_class in matches:
                f.write(f"{src},{tgt},{src_class},{tgt_class}\n")
    except Exception as e:
        logger.error(f"Error saving instance matches to csv: {e}")


def evaluate_hits(candidate_matches, reference_alignments, ks=K_SET):
    """
    Evaluate Hits@K for the candidate matches against the reference alignments.
    Args:
        candidate_matches (list): List of candidate matches as tuples of (source_entity_uri, target_entity_uri)
        reference_alignments (list): List of reference alignments as tuples of (source_entity_uri, target_entity_uri)
        ks (tuple): Tuple of K values to evaluate Hits@K
    Returns:
        dict: Dictionary of Hits@K values
    """
    # prepare reference set for fast lookup
    ref_set = set((str(s), str(t)) for s, t in reference_alignments)

    hits = {k: 0 for k in ks}
    # enumerate candidate matches to get their rank (0-based)
    # for idx, (src, tgt) in enumerate(candidate_matches):
    idx = -1
    candidate_src = ""
    for src, tgt in candidate_matches:
        if src != candidate_src:
            idx = 0
            candidate_src = src
        else:
            idx += 1
        if (str(src), str(tgt)) in ref_set:
            for k in ks:
                if idx < k:
                    hits[k] += 1

    hits_at_k = {}
    for k in sorted(ks):
        ref_alignment_count = len(reference_alignments)
        hits_at_k_value = hits[k] / ref_alignment_count
        logger.info(f"Hits@{k}: {hits_at_k_value:.2f}")
        hits_at_k[k] = hits_at_k_value

    return hits_at_k


def get_bm25_candidates(source_kg_path, target_kg_path, k1, b):
    # Extract entities from source RDF files
    logger.info("Extracting entities from source RDF files...")
    source_class_paragraphs, source_predicate_paragraphs = generate_entity_paragraphs(
        source_kg_path
    )

    # Extract entities from target RDF files
    logger.info("Extracting entities from target RDF files...")
    target_class_paragraphs, target_predicate_paragraphs = generate_entity_paragraphs(
        target_kg_path
    )

    logger.info("Generating Class candidate matches using BM25...")
    class_candidate_matches = get_candidate_matches(
        source_class_paragraphs, target_class_paragraphs, k1, b
    )

    logger.info("Generating Predicate candidate matches using BM25...")
    predicate_candidate_matches = get_candidate_matches(
        source_predicate_paragraphs, target_predicate_paragraphs, k1, b
    )

    logger.info(f"Total Class candidate matches found: {len(class_candidate_matches)}")
    logger.info(
        f"Total Predicate candidate matches found: {len(predicate_candidate_matches)}"
    )

    # save candidate matches to csv files
    logger.info("Saving candidate matches to CSV files...")
    source_kg_name = os.path.basename(source_kg_path).replace(".rdf", "")
    target_kg_name = os.path.basename(target_kg_path).replace(".rdf", "")
    kg_pair = f"{source_kg_name}-{target_kg_name}"
    class_output_file = os.path.join(
        candidates_dir, f"class_candidates_{k1}_{b}_{kg_pair}.csv"
    )
    predicate_output_file = os.path.join(
        candidates_dir, f"predicate_candidates_{k1}_{b}_{kg_pair}.csv"
    )
    save_matches_to_csv(class_candidate_matches, class_output_file)
    save_matches_to_csv(predicate_candidate_matches, predicate_output_file)

    return (class_candidate_matches, predicate_candidate_matches)


def main():
    # get environment configuration for KG pairs
    logger.info("Loading configuration from environment...")
    kg_pair_paths = get_kg_pair_paths()

    # Define BM25 parameter grids - refer bm25_parameter_guide.md for details
    K1_GRID = [0.0, 0.6, 1.2, 1.8, 2.4]
    B_GRID = [0.0, 0.3, 0.75, 1.0]

    # DataFrame to store overall results
    results_df = pd.DataFrame(
        columns=[
            "KG_Pair",
            "k1",
            "b",
            "epsilon",
            "C Hits@1",
            "C Hits@5",
            "C Hits@10",
            "C Hits@20",
            "C Hits@50",
            "P Hits@1",
            "P Hits@5",
            "P Hits@10",
            "P Hits@20",
            "P Hits@50",
            "Total_Time_Seconds",
        ]
    )

    for k1, b in tqdm(
        list(itertools.product(K1_GRID, B_GRID)),
        desc="BM25 Parameter Combinations",
    ):
        logger.info(f"\n=== BM25 Parameters: k1={k1}, b={b} ===\n")

        # process each KG pair sequentially
        for kg_pair, (
            source_path,
            target_path,
            alignment_path,
        ) in kg_pair_paths.items():
            # start time
            start_time = time.time()

            logger.info(f"Processing KG pair: {kg_pair}")
            logger.info("Loading reference alignments...")
            (class_reference_alignments, predicate_reference_alignments, _) = (
                load_reference_alignments(source_path, target_path, alignment_path)
            )

            # Get BM25 candidate matches
            (
                class_candidate_matches,
                predicate_candidate_matches,
            ) = get_bm25_candidates(source_path, target_path, k1, b)

            # Evaluate Hits@K
            logger.info("Evaluating Hits@K for Class candidate matches:")
            class_hits_at_k = evaluate_hits(
                class_candidate_matches, class_reference_alignments
            )

            logger.info("Evaluating Hits@K for Predicate candidate matches:")
            predicate_hits_at_k = evaluate_hits(
                predicate_candidate_matches, predicate_reference_alignments
            )
            # end time
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Total time taken: {total_time:.2f} seconds")
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            {
                                "KG_Pair": kg_pair,
                                "k1": k1,
                                "b": b,
                                "C Hits@1": class_hits_at_k[1],
                                "C Hits@5": class_hits_at_k[5],
                                "C Hits@10": class_hits_at_k[10],
                                "C Hits@20": class_hits_at_k[20],
                                "C Hits@50": class_hits_at_k[50],
                                "P Hits@1": predicate_hits_at_k[1],
                                "P Hits@5": predicate_hits_at_k[5],
                                "P Hits@10": predicate_hits_at_k[10],
                                "P Hits@20": predicate_hits_at_k[20],
                                "P Hits@50": predicate_hits_at_k[50],
                                "Total_Time_Seconds": total_time,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Save overall results to CSV
    results_output_file = os.path.join(results_dir, "bm25_hits_at_k_results.csv")
    results_df.to_csv(results_output_file, index=False)
    logger.info(f"BM25 Hits@K results saved to {results_output_file}")


if __name__ == "__main__":
    main()
