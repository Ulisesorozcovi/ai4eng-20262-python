import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def generar_caso_de_uso_medico() -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    """
    Genera un caso de uso aleatorio para la función preparar_datos.
    
    Returns:
        input_data: Diccionario con el DataFrame y el nombre de la columna objetivo.
        output_data: Tupla (X, y) que representa el resultado esperado.
    """
    # 1. Configuración aleatoria del tamaño del dataset
    n_rows = np.random.randint(5, 15)
    n_cols = np.random.randint(2, 5)
    
    # 2. Generación de datos sintéticos (Características)
    # Creamos una matriz de floats aleatorios
    col_names = [f'feat_{i}' for i in range(n_cols)]
    data = np.random.uniform(10, 100, size=(n_rows, n_cols))
    
    # Introducir NaNs aleatorios (30% de probabilidad por celda)
    mask = np.random.choice([True, False], size=data.shape, p=[0.3, 0.7])
    data[mask] = np.nan
    
    df = pd.DataFrame(data, columns=col_names)
    
    # 3. Generación de la columna objetivo (Target)
    target_col = "target_class"
    df[target_col] = np.random.randint(0, 2, size=n_rows) # Binario 0 o 1
    
    # --- CÁLCULO DEL OUTPUT ESPERADO (Lógica de la misión) ---
    
    # Separar X e y antes de transformar
    X_raw = df.drop(columns=[target_col])
    y_expected = df[target_col].values
    
    # Imputación (Media)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    
    # Escalado (Standard)
    scaler = StandardScaler()
    X_expected = scaler.fit_transform(X_imputed)
    
    # 4. Estructurar Retorno
    input_params = {
        "df": df.copy(),
        "target_col": target_col
    }
    output_expected = (X_expected, y_expected)
    
    return input_params, output_expected

# Ejemplo de uso:
# 1. Ejecutar el generador para obtener un caso de prueba
input_dict, output_tuple = generar_caso_de_uso_medico()

# 2. Desempaquetar los datos para inspección
df_entrada = input_dict['df']
columna_objetivo = input_dict['target_col']
X_esperado, y_esperado = output_tuple

# 3. Mostrar resultados en consola de forma elegante
print("--- DATOS DE ENTRADA (DataFrame Original) ---")
print(df_entrada)
print(f"\nColumna Objetivo seleccionada: {columna_objetivo}")

print("\n--- OUTPUT ESPERADO (Matriz X procesada) ---")
# Mostramos los primeros 5 registros de la matriz escalada e imputada
print(X_esperado[:5]) 

print("\n--- OUTPUT ESPERADO (Vector y) ---")
print(y_esperado)

# 4. Verificación de integridad
print("\n--- VERIFICACIÓN DE DIMENSIONES ---")
print(f"¿Filas coinciden?: {df_entrada.shape[0] == X_esperado.shape[0]}")
print(f"¿Columnas de X correctas?: {X_esperado.shape[1] == df_entrada.shape[1] - 1}")