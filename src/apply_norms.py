import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "apollo_preferences_dataset.csv")
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "learned_agent_weights.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "norms_results.csv")

VALUE_COLUMNS_A = ["v1_A", "v2_A", "v3_A"]
VALUE_COLUMNS_B = ["v1_B", "v2_B", "v3_B"]

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)


def utility(values, weights):
    """
    Utilidad lineal basada en valores.
    """
    return np.dot(weights, values)


def norm_penalty(values):
    """
    Norma simple:
    penaliza alternativas con mal desempeño en el valor 2.

    Interpretación:
    si el valor 2 representa coste, esta norma penaliza rutas con coste alto.
    Como los valores son negativos, valores más bajos implican peor alineación.
    """
    penalty_strength = 0.5
    threshold = -0.2

    if values[1] < threshold:
        return penalty_strength

    return 0.0


def main():
    preferences_df = pd.read_csv(DATA_PATH)
    weights_df = pd.read_csv(WEIGHTS_PATH)

    preferences_df["agent_id"] = preferences_df["agent_id"].astype(str)
    weights_df["agent_id"] = weights_df["agent_id"].astype(str)

    weights_dict = {
        row["agent_id"]: np.array([row["w_v1"], row["w_v2"], row["w_v3"]])
        for _, row in weights_df.iterrows()
    }

    rows = []
    total = 0
    changed = 0

    for _, row in preferences_df.iterrows():
        agent_id = row["agent_id"]

        if agent_id not in weights_dict:
            continue

        weights = weights_dict[agent_id]

        values_A = row[VALUE_COLUMNS_A].values.astype(float)
        values_B = row[VALUE_COLUMNS_B].values.astype(float)

        # Decisión sin norma
        utility_A = utility(values_A, weights)
        utility_B = utility(values_B, weights)

        decision_before = 1 if utility_A > utility_B else 0

        # Decisión con norma
        penalty_A = norm_penalty(values_A)
        penalty_B = norm_penalty(values_B)

        utility_A_norm = utility_A - penalty_A
        utility_B_norm = utility_B - penalty_B

        decision_after = 1 if utility_A_norm > utility_B_norm else 0

        has_changed = int(decision_before != decision_after)

        rows.append({
            "agent_id": agent_id,
            "comparison_id": row["comparison_id"],
            "decision_before": decision_before,
            "decision_after": decision_after,
            "changed": has_changed,
            "utility_A": utility_A,
            "utility_B": utility_B,
            "utility_A_norm": utility_A_norm,
            "utility_B_norm": utility_B_norm,
            "penalty_A": penalty_A,
            "penalty_B": penalty_B,
            "v1_A": values_A[0],
            "v2_A": values_A[1],
            "v3_A": values_A[2],
            "v1_B": values_B[0],
            "v2_B": values_B[1],
            "v3_B": values_B[2],
        })

        total += 1
        changed += has_changed

    if total == 0:
        print("ERROR: no se ha procesado ninguna decisión.")
        print("Comprueba que los agent_id del dataset coinciden con los del archivo de pesos.")
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nResultados de normas guardados en:")
    print(OUTPUT_PATH)

    print("\nTotal decisiones:", total)
    print("Decisiones que cambian por la norma:", changed)
    print("Porcentaje de cambio:", round(100 * changed / total, 2), "%")


if __name__ == "__main__":
    main()