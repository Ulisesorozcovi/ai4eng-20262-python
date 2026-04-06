import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, Any

def generar_caso_de_uso_masas_agua() -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    """
    Genera un caso de uso aleatorio para datos de sensores oceanográficos.
    
    Returns:
        input_data: Diccionario con 'df' y 'target_col'.
        output_data: Tupla (X, y) procesada con RobustScaler e imputación de media.
    """
    # 1. Configuración aleatoria: número de muestras y variables detectadas
    n_muestras = np.random.randint(8, 15)
    variables = ['temp_c', 'salinidad_psu', 'oxigeno_ml_l', 'presion_dbar', 'silicato_umol']
    n_vars = np.random.randint(3, len(variables) + 1)
    cols_x = variables[:n_vars]
    
    # 2. Creación de datos sintéticos con Outliers (típico en oceanografía)
    # Generamos datos normales y sumamos algunos valores extremos aleatorios
    data = np.random.normal(loc=30, scale=10, size=(n_muestras, n_vars))
    outlier_idx = np.random.choice(n_muestras, size=2, replace=False)
    data[outlier_idx] *= 5  # Creamos valores atípicos
    
    # Introducir NaNs (probabilidad del 20%)
    mask = np.random.random(data.shape) < 0.20
    data[mask] = np.nan
    
    df = pd.DataFrame(data, columns=cols_x)
    
    # 3. Variable objetivo: Tipo de masa de agua (Water Mass)
    target_name = "water_mass_type"
    masas = ['NADW', 'AAIW', 'AABW', 'ModeWater']
    df[target_name] = np.random.choice(masas, size=n_muestras)
    
    # --- CÁLCULO DEL OUTPUT ESPERADO ---
    X_raw = df.drop(columns=[target_name])
    y_expected = df[target_name].values
    
    # Imputación (Media)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    
    # Escalado Robusto (basado en cuantiles para manejar los outliers creados)
    scaler = RobustScaler()
    X_expected = scaler.fit_transform(X_imputed)
    
    # 4. Formateo de salida
    input_params = {
        "df": df.copy(),
        "target_col": target_name
    }
    output_expected = (X_expected, y_expected)
    
    return input_params, output_expected

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    input_dict, output_tuple = generar_caso_de_uso_masas_agua()
    
    print("🌊 CASO DE USO OCEANOGRÁFICO GENERADO")
    print("\n[INPUT] Primeras filas del DataFrame (con posibles nulos y outliers):")
    print(input_dict['df'].head())
    
    print(f"\n[INPUT] Columna objetivo: {input_dict['target_col']}")
    
    print("\n[OUTPUT] Matriz X (Tras RobustScaler):")
    # El RobustScaler suele centrar los datos en 0 basándose en la mediana
    print(output_tuple[0][:3]) # Mostramos las primeras 3 filas