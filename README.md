# BM25 for KG Matching - OAEI Study

Companion code/data for paper titled Empirical Evaluation of BM25 Hyperparameters for Candidate Generation in Knowledge Graph Matching" submitted to Spring Nature Computer Science Journal.

This repository contains two main scripts to run a simple BM25-based candidate generator for knowledge graph entity/predicate matching and to analyse results.

## Structure

- bm25_candidate_generator.py  : Extracts verbalizations from RDF KGs and generates BM25 candidate pairs (s, t) saved to `data/candidates/`.
- analyse_bm25_results.py      : Loads candidate CSVs and reference alignments to compute Hits@K metrics and aggregate results (saved to `data/results/`).
- utils.py                     : Helper functions for reading KG pair configs and extracting labels.
- requirements.txt             : Python dependencies.
- bm25_hits_at_k_results.csv   : 100 configs × 5 KG pairs, Columns: KGPair, k1, b, C/P Hits@1/5/10/20/50, runtime

## Usage

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download NLTK data required (once):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

3. Prepare `.env` file with your configuration of the OAEI KG dataset local paths:, e.g.:

```.env
# Path for local ontology files
ONTOLOGIES_BASE_PATH="/path/to/ontologies"

# Path for local reference alignment files
REFERENCES_BASE_PATH="/path/to/references"

# Ontologies stored as a JSON string
ONTOLOGIES_JSON='{
    "marvel": "marvel.rdf",
    "marvelcinematicuniverse": "marvelcinematicuniverse.rdf",
    "memoryalpha": "memoryalpha.rdf",
    "memorybeta": "memorybeta.rdf",
    "starwars": "starwars.rdf",
    "stexpanded": "stexpanded.rdf",
    "swg": "swg.rdf",
    "swtor": "swtor.rdf"
}'

# Gold standard alignments stored as a JSON string (List of Lists/Tuples)
REFERENCE_ALIGNMENTS_JSON='[
    ["marvelcinematicuniverse", "marvel", "marvelcinematicuniverse-marvel.rdf"],
    ["memoryalpha", "memorybeta", "memoryalpha-memorybeta.rdf"],
    ["memoryalpha", "stexpanded", "memoryalpha-stexpanded.rdf"],
    ["starwars", "swg", "starwars-swg.rdf"],
    ["starwars", "swtor", "starwars-swtor.rdf"]
]'
```

4. Run the candidate generator for all pairs configured:

```bash
python bm25_candidate_generator.py
```

5. Analyse results (compute Hits@K from generated candidate CSVs vs reference CSVs):

```bash
python analyse_bm25_results.py --candidates data/candidates --references data/reference_alignments --output data/results
```

Notes

- The scripts expect RDF files in RDF/XML format (`.rdf`). If your files are Turtle, adapt the parsing in the code.
- Steps given here are considering a Mac machine, adjust paths and commands as needed for your environment.
