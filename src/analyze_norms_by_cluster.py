import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMS_PATH = os.path.join(BASE_DIR, "outputs", "norms_results.csv")
CLUSTERS_PATH = os.path.join(BASE_DIR, "outputs", "agent_clusters_comparison.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "norms_by_cluster_summary.csv")

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)


def main():
    norms_df = pd.read_csv(NORMS_PATH)
    clusters_df = pd.read_csv(CLUSTERS_PATH)

    # Nos quedamos con K-Means
    clusters_df = clusters_df[["agent_id", "cluster_kmeans"]]

    norms_df["agent_id"] = norms_df["agent_id"].astype(str)
    clusters_df["agent_id"] = clusters_df["agent_id"].astype(str)

    df = norms_df.merge(clusters_df, on="agent_id")

    print("\n=== Impacto de la norma por cluster ===")

    summary = []

    for cluster_id, group in df.groupby("cluster_kmeans"):
        total = len(group)
        changed = group["changed"].sum()
        percentage = 100 * changed / total

        summary.append({
            "cluster": cluster_id,
            "total_decisions": total,
            "changed": changed,
            "percentage_changed": percentage
        })

        print(f"\nCluster {cluster_id}:")
        print(f"  Total decisiones: {total}")
        print(f"  Cambios: {changed}")
        print(f"  Porcentaje: {percentage:.2f}%")

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("\nResumen:")
    print(summary_df)

    max_cluster = summary_df.loc[
        summary_df["percentage_changed"].idxmax()
    ]

    print("\nCluster más afectado:")
    print(max_cluster)

    print("\nResumen guardado en:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()