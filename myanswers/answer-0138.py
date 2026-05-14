import pandas as pd
import numpy as np

def calcular_entropia_por_grupo(df):
    entropy_results = []

    for g in sorted(df["group"].unique()):
        subset = df[df["group"] == g]
        probs = subset["class_label"].value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        entropy_results.append((g, entropy))

    resultado = pd.DataFrame(entropy_results, columns=["group", "entropy"])

    return resultado