"""
Entity verbalization functions for RDF Knowledge Graph matching.
These functions extract and verbalize entities from RDF graphs for use in candidate generation.
"""

import logging
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef
from utils import extract_label

# Configure logging
logger = logging.getLogger(__name__)


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


def triple_graph_verbalization(g, uri_str):
    """
    Verbalize an entity given its URI by extracting all triples where it is the subject,
    predicate or object, creating a graph neighborhood description.
    This approach builds entity descriptions by combining RDF triples from the entity's
    neighborhood in the graph, including relationships where the entity appears as
    subject, predicate, or object.

    Args:
        g (Graph): The RDF graph
        uri_str (str): The URI to verbalize
    Returns:
        str: The verbalized form of the URI as a paragraph describing its neighborhood in the graph
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
            class_corpus[entity] = triple_graph_verbalization(g, entity)
        elif entity_type == "Predicate":
            predicate_corpus[entity] = triple_graph_verbalization(g, entity)
        else:
            logger.warning(f"Entity {entity} of type {entity_type} skipped.")

    logger.info(f"Total number of Class entities: {len(class_corpus)}")
    logger.info(f"Total number of Predicate entities: {len(predicate_corpus)}")
    return class_corpus, predicate_corpus
