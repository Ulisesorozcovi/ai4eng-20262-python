from sklearn.ensemble import RandomForestClassifier

def calcular_importancia_caracteristicas(df_empleados):
    df_clean = df_empleados.dropna()

    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]

    y = (y > 0.5).astype(int)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    return modelo.feature_importances_