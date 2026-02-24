import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any

def generar_caso_de_uso_preparar_datos() -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    """
    Genera un caso de uso aleatorio para la función preparar_datos en un contexto ambiental.
    
    Returns:
        input_data: Diccionario con {'df': pd.DataFrame, 'target_col': str}
        output_data: Tupla (X, y) con los datos procesados (imputados y escalados [0,1]).
    """
    # 1. Configuración aleatoria del escenario (sensores y días de medición)
    n_dias = np.random.randint(6, 12)
    sensores = ['pm25_µg', 'no2_ppb', 'o3_ppm', 'so2_ppb']
    n_sensores = np.random.randint(2, len(sensores) + 1)
    columnas_seleccionadas = sensores[:n_sensores]
    
    # 2. Generación de lecturas de sensores (con valores realistas)
    data = np.random.uniform(0.1, 100.0, size=(n_dias, n_sensores))
    
    # Introducir fallos de sensor (NaNs) de forma aleatoria (25% de probabilidad)
    mask = np.random.choice([True, False], size=data.shape, p=[0.25, 0.75])
    data[mask] = np.nan
    
    df = pd.DataFrame(data, columns=columnas_seleccionadas)
    
    # 3. Variable objetivo: Categoría de calidad del aire
    target_name = "calidad_aire"
    categorias = ['Excelente', 'Moderado', 'Peligroso']
    df[target_name] = np.random.choice(categorias, size=n_dias)
    
    # --- PROCESAMIENTO PARA EL OUTPUT ESPERADO ---
    X_raw = df.drop(columns=[target_name])
    y_expected = df[target_name].values
    
    # Imputación por Mediana
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X_raw)
    
    # Escalado Min-Max [0, 1]
    scaler = MinMaxScaler()
    X_expected = scaler.fit_transform(X_filled)
    
    # 4. Empaquetado de resultados
    input_params = {
        "df": df.copy(),
        "target_col": target_name
    }
    output_tuple = (X_expected, y_expected)
    
    return input_params, output_tuple

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Generar el par entrada/salida
    entrada, salida_esperada = generar_caso_de_uso_preparar_datos()
    
    print("🚀 NUEVO CASO DE PRUEBA GENERADO")
    print("\n[INPUT] DataFrame Original (con nulos):")
    print(entrada['df'])
    print(f"\n[INPUT] Columna objetivo: {entrada['target_col']}")
    
    print("\n[OUTPUT] Matriz X (Imputada con mediana y Escalada 0-1):")
    # Redondeamos para visualización limpia
    print(np.round(salida_esperada[0], 4)) 
    
    print("\n[OUTPUT] Vector y (Etiquetas):")
    print(salida_esperada[1])