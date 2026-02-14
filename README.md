# BM25 for KG Matching - OAEI Study

Companion code/data for paper titled Empirical Evaluation of BM25 Hyperparameters for Candidate Generation in Knowledge Graph Matching" submitted to Spring Nature Computer Science Journal.

## Data
- `data/bm25_hits_at_k_results.csv`: 100 configs × 5 KG pairs
- Columns: KGPair, k1, b, C/P Hits@1/5/10/20/50, runtime

## Reproduce Charts
```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
