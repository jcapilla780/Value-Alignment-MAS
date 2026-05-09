import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "apollo_preferences_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "learned_agent_weights.csv")

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)


def load_dataset(path):
    return pd.read_csv(path)


def build_pairwise_dataframe(df):
    df = df.copy()

    # choice = 1 si gana A, choice = 2 si gana B
    df["choice_A"] = (df["choice"] == 1).astype(int)

    df["diff_v1"] = df["v1_A"] - df["v1_B"]
    df["diff_v2"] = df["v2_A"] - df["v2_B"]
    df["diff_v3"] = df["v3_A"] - df["v3_B"]

    return df


def learn_weights_for_agent(agent_df):
    value_cols = ["diff_v1", "diff_v2", "diff_v3"]

    X = agent_df[value_cols].values
    y = agent_df["choice_A"].values

    if len(np.unique(y)) < 2:
        return None

    model = LogisticRegression(
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000
    )

    model.fit(X, y)

    weights = model.coef_[0]

    # Para interpretabilidad: pesos positivos y normalizados
    weights = np.maximum(weights, 0)

    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    return weights


def main():
    raw_df = load_dataset(DATA_PATH)
    df = build_pairwise_dataframe(raw_df)

    print("Dataset cargado")
    print("Número de comparaciones:", len(df))
    print("Número de agentes:", df["agent_id"].nunique())
    print(df.head())

    learned_rows = []

    for agent_id, agent_df in df.groupby("agent_id"):
        weights = learn_weights_for_agent(agent_df)

        if weights is None:
            continue

        learned_rows.append({
            "agent_id": str(agent_id),
            "w_v1": weights[0],
            "w_v2": weights[1],
            "w_v3": weights[2],
            "n_preferences": len(agent_df)
        })

    weights_df = pd.DataFrame(learned_rows)
    weights_df.to_csv(OUTPUT_PATH, index=False)

    print("\nPesos aprendidos guardados en:")
    print(OUTPUT_PATH)

    print("\nPrimeros pesos aprendidos:")
    print(weights_df.head())

    print("\nResumen:")
    print(weights_df.describe())


if __name__ == "__main__":
    main()