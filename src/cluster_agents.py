import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "outputs", "learned_agent_weights.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "agent_clusters_comparison.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "outputs", "clustering_summary.csv")

VALUE_COLUMNS = ["w_v1", "w_v2", "w_v3"]
N_CLUSTERS = 3
RANDOM_STATE = 42

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)


def compute_cluster_profiles(df, label_col):
    profiles = (
        df.groupby(label_col)[VALUE_COLUMNS]
        .mean()
        .reset_index()
        .sort_values(label_col)
    )

    sizes = df[label_col].value_counts().sort_index()
    profiles["cluster_size"] = profiles[label_col].map(sizes)

    return profiles


def print_profiles(method_name, profiles, label_col):
    print(f"\n=== {method_name} ===")
    print("\nTamaño y centroides de clusters:")

    for _, row in profiles.iterrows():
        cluster_id = int(row[label_col])
        print(f"\nCluster {cluster_id}:")
        print(f"  tamaño: {int(row['cluster_size'])}")
        print(f"  w_v1: {row['w_v1']:.3f}")
        print(f"  w_v2: {row['w_v2']:.3f}")
        print(f"  w_v3: {row['w_v3']:.3f}")

        dominant_value = VALUE_COLUMNS[
            int(np.argmax([row["w_v1"], row["w_v2"], row["w_v3"]]))
        ]
        print(f"  valor dominante: {dominant_value}")


def main():
    df = pd.read_csv(INPUT_PATH)

    df["agent_id"] = df["agent_id"].astype(str)

    X = df[VALUE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results_summary = []

    # -------------------------
    # 1. K-Means
    # -------------------------
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

    sil_kmeans = silhouette_score(X_scaled, df["cluster_kmeans"])

    profiles_kmeans = compute_cluster_profiles(df, "cluster_kmeans")
    print_profiles("K-Means", profiles_kmeans, "cluster_kmeans")

    results_summary.append({
        "method": "K-Means",
        "silhouette_score": sil_kmeans,
        "n_clusters": N_CLUSTERS
    })

    # -------------------------
    # 2. Clustering jerárquico
    # -------------------------
    hierarchical = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="ward"
    )

    df["cluster_hierarchical"] = hierarchical.fit_predict(X_scaled)

    sil_hierarchical = silhouette_score(X_scaled, df["cluster_hierarchical"])

    profiles_hierarchical = compute_cluster_profiles(df, "cluster_hierarchical")
    print_profiles("Clustering jerárquico", profiles_hierarchical, "cluster_hierarchical")

    results_summary.append({
        "method": "Hierarchical",
        "silhouette_score": sil_hierarchical,
        "n_clusters": N_CLUSTERS
    })

    # -------------------------
    # 3. Gaussian Mixture Model
    # -------------------------
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        random_state=RANDOM_STATE
    )

    df["cluster_gmm"] = gmm.fit_predict(X_scaled)

    sil_gmm = silhouette_score(X_scaled, df["cluster_gmm"])

    profiles_gmm = compute_cluster_profiles(df, "cluster_gmm")
    print_profiles("Gaussian Mixture Model", profiles_gmm, "cluster_gmm")

    results_summary.append({
        "method": "GMM",
        "silhouette_score": sil_gmm,
        "n_clusters": N_CLUSTERS
    })

    # -------------------------
    # Comparación entre métodos
    # -------------------------
    print("\n=== Comparación cuantitativa ===")

    summary_df = pd.DataFrame(results_summary)
    print(summary_df)

    print("\nAdjusted Rand Index entre métodos:")
    ari_k_h = adjusted_rand_score(df["cluster_kmeans"], df["cluster_hierarchical"])
    ari_k_g = adjusted_rand_score(df["cluster_kmeans"], df["cluster_gmm"])
    ari_h_g = adjusted_rand_score(df["cluster_hierarchical"], df["cluster_gmm"])

    print(f"K-Means vs Jerárquico: {ari_k_h:.3f}")
    print(f"K-Means vs GMM:        {ari_k_g:.3f}")
    print(f"Jerárquico vs GMM:     {ari_h_g:.3f}")

    summary_df["ari_vs_kmeans"] = [
        1.0,
        ari_k_h,
        ari_k_g
    ]

    df.to_csv(OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nArchivos guardados:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()