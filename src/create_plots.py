import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(GRAPHS_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(OUTPUTS_DIR, "learned_agent_weights.csv")
CLUSTERS_PATH = os.path.join(OUTPUTS_DIR, "agent_clusters_comparison.csv")
CLUSTERING_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "clustering_summary.csv")
NORMS_PATH = os.path.join(OUTPUTS_DIR, "norms_results.csv")
NORMS_BY_CLUSTER_PATH = os.path.join(OUTPUTS_DIR, "norms_by_cluster_summary.csv")

VALUE_COLUMNS = ["w_v1", "w_v2", "w_v3"]


def save_plot(filename):
    path = os.path.join(GRAPHS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Gráfico guardado: {path}")


def plot_weights_distribution(weights_df):
    plt.figure(figsize=(8, 5))
    weights_df[VALUE_COLUMNS].boxplot()
    plt.title("Distribución de pesos de valores aprendidos")
    plt.ylabel("Peso aprendido")
    plt.xlabel("Valor")
    save_plot("weights_distribution_boxplot.png")


def plot_agents_scatter(clusters_df):
    plt.figure(figsize=(7, 5))

    for cluster_id, group in clusters_df.groupby("cluster_kmeans"):
        plt.scatter(
            group["w_v1"],
            group["w_v3"],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )

    plt.title("Agentes según pesos aprendidos")
    plt.xlabel("Peso valor 1")
    plt.ylabel("Peso valor 3")
    plt.legend()
    save_plot("agents_scatter_kmeans.png")


def plot_cluster_centroids(clusters_df):
    centroids = (
        clusters_df
        .groupby("cluster_kmeans")[VALUE_COLUMNS]
        .mean()
        .sort_index()
    )

    centroids.plot(kind="bar", figsize=(8, 5))

    plt.title("Centroides de clusters K-Means")
    plt.xlabel("Cluster")
    plt.ylabel("Peso medio")
    plt.xticks(rotation=0)
    plt.legend(title="Valores")
    save_plot("cluster_centroids_kmeans.png")


def plot_clustering_silhouette(summary_df):
    plt.figure(figsize=(7, 5))

    plt.bar(summary_df["method"], summary_df["silhouette_score"])

    plt.title("Comparación de métodos de clustering")
    plt.xlabel("Método")
    plt.ylabel("Silhouette score")
    save_plot("clustering_silhouette_comparison.png")


def plot_norm_global_impact(norms_df):
    counts = norms_df["changed"].value_counts().sort_index()

    labels = ["No cambia", "Cambia"]
    values = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)

    plt.title("Impacto global de la norma")
    plt.ylabel("Número de decisiones")
    save_plot("norm_global_impact.png")


def plot_norm_impact_by_cluster(norms_cluster_df):
    plt.figure(figsize=(7, 5))

    plt.bar(
        norms_cluster_df["cluster"].astype(str),
        norms_cluster_df["percentage_changed"]
    )

    plt.title("Impacto de la norma por cluster")
    plt.xlabel("Cluster K-Means")
    plt.ylabel("Decisiones modificadas (%)")
    save_plot("norm_impact_by_cluster.png")


def plot_norm_decision_transitions(norms_df):
    transitions = norms_df.groupby(
        ["decision_before", "decision_after"]
    ).size().reset_index(name="count")

    labels = []
    values = []

    for _, row in transitions.iterrows():
        before = "A" if row["decision_before"] == 1 else "B"
        after = "A" if row["decision_after"] == 1 else "B"

        labels.append(f"{before} → {after}")
        values.append(row["count"])

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)

    plt.title("Transiciones de decisión tras aplicar normas")
    plt.xlabel("Cambio de decisión")
    plt.ylabel("Número de decisiones")
    save_plot("norm_decision_transitions.png")


def main():
    weights_df = pd.read_csv(WEIGHTS_PATH)
    clusters_df = pd.read_csv(CLUSTERS_PATH)
    clustering_summary_df = pd.read_csv(CLUSTERING_SUMMARY_PATH)
    norms_df = pd.read_csv(NORMS_PATH)
    norms_cluster_df = pd.read_csv(NORMS_BY_CLUSTER_PATH)

    plot_weights_distribution(weights_df)
    plot_agents_scatter(clusters_df)
    plot_cluster_centroids(clusters_df)
    plot_clustering_silhouette(clustering_summary_df)
    plot_norm_global_impact(norms_df)
    plot_norm_impact_by_cluster(norms_cluster_df)
    plot_norm_decision_transitions(norms_df)

    print("\nTodos los gráficos se han generado correctamente.")


if __name__ == "__main__":
    main()