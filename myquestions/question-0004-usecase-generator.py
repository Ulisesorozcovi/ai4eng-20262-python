import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def generar_caso_de_uso_preparar_datos() -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    """
    Genera un caso de uso aleatorio para la gestión de logística portuaria.
    
    Returns:
        input_data: Diccionario con el DataFrame y el nombre de la columna objetivo.
        output_data: Tupla (X, y) con datos imputados (media) y escalados (estándar).
    """
    # 1. Parámetros aleatorios del puerto
    n_buques = np.random.randint(10, 20)
    indicadores = ['tonelaje_carga', 'n_movimientos_grua', 'velocidad_viento_nudos', 'n_camiones_espera']
    n_indicadores = np.random.randint(2, len(indicadores) + 1)
    cols_x = indicadores[:n_indicadores]
    
    # 2. Generación de datos sintéticos
    # Usamos distribuciones normales para simular métricas operativas
    data = np.random.uniform(5, 500, size=(n_buques, n_indicadores))
    
    # Introducir nulos aleatorios (fallos en sensores del puerto - 15% probabilidad)
    mask = np.random.rand(*data.shape) < 0.15
    data[mask] = np.nan
    
    df = pd.DataFrame(data, columns=cols_x)
    
    # 3. Variable objetivo: Tiempo de respuesta del muelle (horas)
    target_name = "turnaround_time_hrs"
    df[target_name] = np.random.uniform(10, 48, size=n_buques)
    
    # --- PROCESAMIENTO PARA EL OUTPUT ESPERADO ---
    X_raw = df.drop(columns=[target_name])
    y_expected = df[target_name].values
    
    # Imputación por Media (Standard para logística)
    imputer = SimpleImputer(strategy='mean')
    X_filled = imputer.fit_transform(X_raw)
    
    # Escalado Estándar (Media 0, Desviación 1)
    scaler = StandardScaler()
    X_expected = scaler.fit_transform(X_filled)
    
    # 4. Empaquetado
    input_params = {
        "df": df.copy(),
        "target_col": target_name
    }
    output_tuple = (X_expected, y_expected)
    
    return input_params, output_tuple

# --- EJEMPLO DE USO Y VISUALIZACIÓN ---
if __name__ == "__main__":
    # Generamos el caso
    entrada_puerto, salida_esperada = generar_caso_de_uso_preparar_datos()
    
    print("🚢 GENERADOR DE LOGÍSTICA PORTUARIA ACTIVADO")
    print("-" * 50)
    print("\n[INPUT] Diccionario de argumentos:")
    print(f"Claves: {list(entrada_puerto.keys())}")
    print("\n[INPUT] Muestra del DataFrame generado (df):")
    print(entrada_puerto['df'].head())
    
    print("\n" + "="*50)
    print("[OUTPUT] Matriz X (Procesada):")
    # Mostramos los primeros 3 registros
    print(salida_esperada[0][:3])
    
    print("\n[OUTPUT] Vector y (Target):")
    print(salida_esperada[1][:3])
    print("-" * 50)