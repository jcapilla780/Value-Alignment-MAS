import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "apollo_preferences_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "learning_evaluation_summary.csv")

VALUE_COLUMNS = ["diff_v1", "diff_v2", "diff_v3"]

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)


def prepare_dataset(df):
    df = df.copy()

    df["agent_id"] = df["agent_id"].astype(str)

    # choice = 1 si gana A, choice = 2 si gana B
    df["choice_A"] = (df["choice"] == 1).astype(int)

    df["diff_v1"] = df["v1_A"] - df["v1_B"]
    df["diff_v2"] = df["v2_A"] - df["v2_B"]
    df["diff_v3"] = df["v3_A"] - df["v3_B"]

    return df


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def evaluate_agent_leave_one_out(agent_df):
    X = agent_df[VALUE_COLUMNS].values
    y = agent_df["choice_A"].values

    # Si todas las preferencias son iguales, no se puede entrenar regresión logística
    if len(np.unique(y)) < 2:
        return None

    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Puede ocurrir que al dejar una observación fuera,
        # el conjunto de entrenamiento tenga una sola clase.
        if len(np.unique(y_train)) < 2:
            continue

        model = train_logistic_regression(X_train, y_train)

        pred = model.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)

    if len(y_true) == 0:
        return None

    accuracy = accuracy_score(y_true, y_pred)

    # Baseline por agente: predecir siempre la clase mayoritaria del agente
    majority_class = pd.Series(y).mode()[0]
    baseline_pred = np.full(len(y), majority_class)
    baseline_accuracy = accuracy_score(y, baseline_pred)

    return {
        "agent_id": agent_df["agent_id"].iloc[0],
        "n_preferences": len(agent_df),
        "n_evaluated_preferences": len(y_true),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy
    }


def main():
    raw_df = pd.read_csv(DATA_PATH)
    df = prepare_dataset(raw_df)

    results = []

    for agent_id, agent_df in df.groupby("agent_id"):
        result = evaluate_agent_leave_one_out(agent_df)

        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)

    summary = {
        "n_agents_evaluated": results_df["agent_id"].nunique(),
        "total_evaluated_preferences": results_df["n_evaluated_preferences"].sum(),
        "mean_accuracy": results_df["accuracy"].mean(),
        "median_accuracy": results_df["accuracy"].median(),
        "mean_baseline_accuracy": results_df["baseline_accuracy"].mean(),
        "median_baseline_accuracy": results_df["baseline_accuracy"].median(),
        "min_accuracy": results_df["accuracy"].min(),
        "max_accuracy": results_df["accuracy"].max()
    }

    summary_df = pd.DataFrame([summary])

    output_path = OUTPUT_PATH
    detailed_output_path = os.path.join(
        BASE_DIR,
        "outputs",
        "learning_evaluation_by_agent.csv"
    )

    summary_df.to_csv(output_path, index=False)
    results_df.to_csv(detailed_output_path, index=False)

    print("\n=== Evaluación Leave-One-Out del aprendizaje ===")
    print(f"Agentes evaluados: {summary['n_agents_evaluated']}")
    print(f"Preferencias evaluadas: {summary['total_evaluated_preferences']}")
    print(f"Accuracy medio: {summary['mean_accuracy']:.3f}")
    print(f"Accuracy mediano: {summary['median_accuracy']:.3f}")
    print(f"Baseline medio: {summary['mean_baseline_accuracy']:.3f}")
    print(f"Baseline mediano: {summary['median_baseline_accuracy']:.3f}")
    print(f"Accuracy mínimo: {summary['min_accuracy']:.3f}")
    print(f"Accuracy máximo: {summary['max_accuracy']:.3f}")

    print("\nArchivos guardados:")
    print(output_path)
    print(detailed_output_path)


if __name__ == "__main__":
    main()