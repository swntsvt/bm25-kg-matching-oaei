import os
import pandas as pd
import argparse


def load_results(file_path):
    """Load results from CSV file"""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return None
    return pd.read_csv(file_path)


def compare_approaches(bm25_results, tfidf_results):
    """Compare BM25 and TF-IDF results"""

    # Create a summary of the comparison
    comparison_results = []

    # Get unique KG pairs
    kg_pairs = set(bm25_results["KG_Pair"].unique()) & set(
        tfidf_results["KG_Pair"].unique()
    )

    # Define the metrics to compare
    class_metrics = ["C Hits@1", "C Hits@5", "C Hits@10", "C Hits@20", "C Hits@50"]
    predicate_metrics = ["P Hits@1", "P Hits@5", "P Hits@10", "P Hits@20", "P Hits@50"]

    for kg_pair in kg_pairs:
        # Get BM25 results for this KG pair
        bm25_pair_data = bm25_results[bm25_results["KG_Pair"] == kg_pair]

        # Get TF-IDF results for this KG pair
        tfidf_pair_data = tfidf_results[tfidf_results["KG_Pair"] == kg_pair]

        # For each metric, compare the best performing approach
        for metric in class_metrics:
            bm25_value = bm25_pair_data[metric].max() if len(bm25_pair_data) > 0 else 0
            tfidf_value = (
                tfidf_pair_data[metric].max() if len(tfidf_pair_data) > 0 else 0
            )

            if bm25_value > tfidf_value:
                winner = "BM25"
                advantage = bm25_value - tfidf_value
            elif tfidf_value > bm25_value:
                winner = "TF-IDF"
                advantage = tfidf_value - bm25_value
            else:
                winner = "Tie"
                advantage = 0

            comparison_results.append(
                {
                    "KG_Pair": kg_pair,
                    "Metric": metric,
                    "BM25_Value": bm25_value,
                    "TF-IDF_Value": tfidf_value,
                    "Winner": winner,
                    "Advantage": advantage,
                }
            )

        for metric in predicate_metrics:
            bm25_value = bm25_pair_data[metric].max() if len(bm25_pair_data) > 0 else 0
            tfidf_value = (
                tfidf_pair_data[metric].max() if len(tfidf_pair_data) > 0 else 0
            )

            if bm25_value > tfidf_value:
                winner = "BM25"
                advantage = bm25_value - tfidf_value
            elif tfidf_value > bm25_value:
                winner = "TF-IDF"
                advantage = tfidf_value - bm25_value
            else:
                winner = "Tie"
                advantage = 0

            comparison_results.append(
                {
                    "KG_Pair": kg_pair,
                    "Metric": metric,
                    "BM25_Value": bm25_value,
                    "TF-IDF_Value": tfidf_value,
                    "Winner": winner,
                    "Advantage": advantage,
                }
            )

    return pd.DataFrame(comparison_results)


def main():
    parser = argparse.ArgumentParser(description="Compare BM25 and TF-IDF results")
    parser.add_argument(
        "--bm25_results", required=True, help="Path to BM25 results CSV file"
    )
    parser.add_argument(
        "--tfidf_results", required=True, help="Path to TF-IDF results CSV file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for comparison results"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load results
    bm25_results = load_results(args.bm25_results)
    tfidf_results = load_results(args.tfidf_results)

    if bm25_results is None or tfidf_results is None:
        print("Error loading results")
        return

    # Compare approaches
    comparison_df = compare_approaches(bm25_results, tfidf_results)

    # Save comparison results
    output_file = os.path.join(args.output, "bm25_vs_tfidf_comparison.csv")
    comparison_df.to_csv(output_file, index=False)

    print(f"Comparison results saved to {output_file}")

    # Print summary
    print("\n=== Comparison Summary ===")
    print(f"Total KG pairs compared: {len(comparison_df['KG_Pair'].unique())}")
    print(f"Total metrics compared: {len(comparison_df)}")

    # Count winners
    winner_counts = comparison_df["Winner"].value_counts()
    print("\n=== Winner Counts ===")
    for winner, count in winner_counts.items():
        print(f"{winner}: {count}")

    # Show top performing approaches for each metric
    print("\n=== Top Performers by Metric ===")
    for metric in sorted(comparison_df["Metric"].unique()):
        top_performer = (
            comparison_df[comparison_df["Metric"] == metric]
            .sort_values("BM25_Value", ascending=False)
            .iloc[0]
        )
        print(
            f"{metric}: {top_performer['Winner']} ({top_performer['BM25_Value']:.4f} vs {top_performer['TF-IDF_Value']:.4f})"
        )


if __name__ == "__main__":
    main()
