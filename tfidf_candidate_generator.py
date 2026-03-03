import itertools
import logging
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rdflib import Graph
from tqdm import tqdm

from utils import get_kg_pair_paths
from text_verbalization.entity_verbalizer import generate_entity_paragraphs
from text_verbalization.preprocessing import preprocess_corpus

# Hits@K values to evaluate
K_SET = (1, 5, 10, 20, 50)
K = 50

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
        logging.FileHandler(os.path.join(logs_dir, "tfidf_candidate_generator.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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


def get_candidate_matches(
    source_paragraphs, target_paragraphs, max_features=None, ngram_range=(1, 2)
):
    """
    Get candidate matches using TF-IDF and cosine similarity

    Args:
        source_paragraphs (dict): Dictionary of source entity URIs to their verbalized paragraphs
        target_paragraphs (dict): Dictionary of target entity URIs to their verbalized paragraphs
        max_features (int): Maximum number of features for TF-IDF vectorization
        ngram_range (tuple): N-gram range for TF-IDF vectorization

    Returns:
        list: List of candidate matches as tuples of (source_entity_uri, target_entity_uri)
    """
    target_ids = list(target_paragraphs.keys())
    target_corpus = list(target_paragraphs.values())

    # Preprocess target corpus using NLTK-based preprocessing (same as BM25)
    logger.info("Preprocessing target corpus...")
    target_corpus_tokens = preprocess_corpus(target_corpus)
    target_corpus_processed = [" ".join(tokens) for tokens in target_corpus_tokens]

    # Create TF-IDF vectorizer
    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
        token_pattern=r"\b[a-zA-Z]{2,}\b",  # Only words with 2+ characters
    )

    # Fit and transform target corpus
    logger.info("Fitting TF-IDF vectorizer on target corpus...")
    target_tfidf_matrix = vectorizer.fit_transform(target_corpus_processed)

    candidate_matches = []

    # Process source paragraphs
    for source_uri_str, source_paragraph in source_paragraphs.items():
        # Preprocess source paragraph using same pipeline
        source_tokens = preprocess_corpus([source_paragraph])
        source_processed = [" ".join(tokens) for tokens in source_tokens]

        # Transform source paragraph to TF-IDF vector
        source_tfidf_vector = vectorizer.transform(source_processed)

        # Compute cosine similarity with all target vectors
        similarities = cosine_similarity(
            source_tfidf_vector, target_tfidf_matrix
        ).flatten()

        # Get top K matches
        top_indices = similarities.argsort()[::-1][:K]

        for i in range(len(top_indices)):
            target_idx = top_indices[i]
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


def get_tfidf_candidates(
    source_kg_path, target_kg_path, max_features=None, ngram_range=(1, 2)
):
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

    logger.info("Generating Class candidate matches using TF-IDF...")
    class_candidate_matches = get_candidate_matches(
        source_class_paragraphs, target_class_paragraphs, max_features, ngram_range
    )

    logger.info("Generating Predicate candidate matches using TF-IDF...")
    predicate_candidate_matches = get_candidate_matches(
        source_predicate_paragraphs,
        target_predicate_paragraphs,
        max_features,
        ngram_range,
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
        candidates_dir,
        f"class_candidates_tfidf_{max_features}_{ngram_range}_{kg_pair}.csv",
    )
    predicate_output_file = os.path.join(
        candidates_dir,
        f"predicate_candidates_tfidf_{max_features}_{ngram_range}_{kg_pair}.csv",
    )
    save_matches_to_csv(class_candidate_matches, class_output_file)
    save_matches_to_csv(predicate_candidate_matches, predicate_output_file)

    return (class_candidate_matches, predicate_candidate_matches)


def main():
    # get environment configuration for KG pairs
    logger.info("Loading configuration from environment...")
    kg_pair_paths = get_kg_pair_paths()

    # Define TF-IDF parameter grids - for comparison with BM25
    MAX_FEATURES_GRID = [1000, 5000, 10000]  # Different feature limits
    NGRAM_RANGE_GRID = [(1, 1), (1, 2), (1, 3)]  # Different n-gram ranges

    # DataFrame to store overall results
    results_df = pd.DataFrame(
        columns=[
            "KG_Pair",
            "max_features",
            "ngram_range",
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

    for max_features, ngram_range in tqdm(
        list(itertools.product(MAX_FEATURES_GRID, NGRAM_RANGE_GRID)),
        desc="TF-IDF Parameter Combinations",
    ):
        logger.info(
            f"\n=== TF-IDF Parameters: max_features={max_features}, ngram_range={ngram_range} ===\n"
        )

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

            # Get TF-IDF candidate matches
            (
                class_candidate_matches,
                predicate_candidate_matches,
            ) = get_tfidf_candidates(
                source_path, target_path, max_features, ngram_range
            )

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
                                "max_features": max_features,
                                "ngram_range": str(ngram_range),
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
    results_output_file = os.path.join(
        results_dir, "tfidf_hits_at_k_results_02Mar2026.csv"
    )
    results_df.to_csv(results_output_file, index=False)
    logger.info(f"TF-IDF Hits@K results saved to {results_output_file}")


if __name__ == "__main__":
    main()
