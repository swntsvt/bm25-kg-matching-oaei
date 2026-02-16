import os
import argparse
import pandas as pd


def compute_hits_at_k(candidates_df, references_set, ks=(1, 5, 10, 20, 50)):
    # candidates_df expected columns: source,target in ranked order grouped by source
    hits = {k: 0 for k in ks}
    total_refs = len(references_set) if references_set else 1
    current_src = None
    rank = 0
    for idx, row in candidates_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        if src != current_src:
            current_src = src
            rank = 0
        else:
            rank += 1
        if (src, tgt) in references_set:
            for k in ks:
                if rank < k:
                    hits[k] += 1
    return {k: hits[k] / total_refs for k in ks}


def load_reference_csv(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return set()
    return set((str(r[0]), str(r[1])) for r in df.iloc[:, :2].values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", required=True, help="Folder with candidate CSV files"
    )
    parser.add_argument(
        "--references", required=True, help="Folder with reference CSV files"
    )
    parser.add_argument(
        "--output", required=True, help="Output folder for aggregated results"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = []
    for fname in os.listdir(args.candidates):
        if not fname.endswith(".csv"):
            continue
        cand_path = os.path.join(args.candidates, fname)
        candidates_df = pd.read_csv(cand_path)
        # try to find matching reference by heuristics
        ref_name = fname.replace("class_candidates", "class_reference").replace(
            "predicate_candidates", "predicate_reference"
        )
        ref_path = os.path.join(args.references, ref_name)
        refs = load_reference_csv(ref_path)
        hits = compute_hits_at_k(candidates_df, refs)
        row = {"candidates_file": fname, **{f"Hits@{k}": v for k, v in hits.items()}}
        results.append(row)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(os.path.join(args.output, "aggregate_hits_at_k.csv"), index=False)


if __name__ == "__main__":
    main()
