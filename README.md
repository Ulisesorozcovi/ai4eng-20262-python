# Modelos y simulacion 1 / 2026-2 - Fase 1: Generación de Casos de Uso

Este repositorio contiene la entrega de la **Fase 1** y la **Fase 2** para el curso de Inteligencia Artificial aplicada a la Ingeniería. El objetivo principal es el diseño de problemas técnicos y la implementación de generadores de datos sintéticos para validar algoritmos de preprocesamiento.

## 👤 Información del Estudiante
* **Nombre:** Ulises Orozco Villegas
* **Correo Institucional:** Ulises.orozco@udea.edu.co
* **Institución:** Universidad de Antioquia

---

## 📂 Estructura del Repositorio

El proyecto se organiza siguiendo una jerarquía estricta para separar la lógica de las preguntas creadas en la Fase 1 y las respuestas desarrolladas en la Fase 2:

```text
.
├── myquestions/                # Contenedor de retos y generadores
│   ├── question-0001.txt       # Problema: Preprocesamiento Médico (KNN/Logística)
│   ├── question-0001-usecase-generator.py
│   ├── question-0002.txt       # Problema: Calidad del Aire (Min-Max Scaling)
│   ├── question-0002-usecase-generator.py
│   ├── question-0003.txt       # Problema: Oceanografía (Robust Scaling)
│   ├── question-0003-usecase-generator.py
│   ├── question-0004.txt       # Problema: Logtistica portuaria (Port Throughput)
│   └── question-0004-usecase-generator.py
├── myanswers/                  # Soluciones desarrolladas en la Fase 2
│   ├── answer-0138.py          # Cálculo de entropía de Shannon por grupo
│   ├── answer-0288.py          # Importancia de características con Random Forest
│   ├── answer-0452.py          # Correlación de rangos con Spearman
│   ├── answer-0622.py          # Vectorización TF-IDF de sentencias judiciales
│   └── diagnostico.txt         # Observaciones sobre validación de soluciones
└── README.md                   # Documentación principal
```
---

## 🛠️ Descripción de los Módulos

1. Generación de Datos (Fase 1)
Cada archivo question-XXXX.txt plantea un escenario de ingeniería donde los datos requieren tratamiento. Los scripts de Python asociados (*-generator.py) implementan la lógica necesaria para Simulación de Errores, Generación Sintétic y Cálculo de Ground Truth.

2. Técnicas de Preprocesamiento Implementadas

En esta fase se han cubierto las siguientes técnicas críticas:
- Imputación: Uso de media y mediana mediante SimpleImputer.
- Escalado: Implementación de StandardScaler, MinMaxScaler y RobustScaler (especialmente útil para datos con outliers).

3. Soluciones Implementadas (Fase 2)

En la Fase 2 se desarrollaron las funciones solución correspondientes a las preguntas asignadas por el curso. Cada archivo `answer-XXXX.py` contiene una única función principal compatible con el validador automático.

| Archivo | Función implementada | Descripción |
|---|---|---|
| `answer-0138.py` | `calcular_entropia_por_grupo(df)` | Calcula la entropía de Shannon para cada grupo a partir de la distribución de etiquetas de clase. |
| `answer-0288.py` | `calcular_importancia_caracteristicas(df_empleados)` | Limpia valores nulos, entrena un `RandomForestClassifier` y devuelve la importancia de las características. |
| `answer-0452.py` | `analizar_correlacion_rangos(df)` | Calcula la matriz de correlación de Spearman para variables numéricas. |
| `answer-0622.py` | `vectorizar_sentencias(df)` | Aplica `TfidfVectorizer` a fragmentos de texto jurídico y devuelve la matriz TF-IDF junto con el vocabulario. |

---

## 🚀 Requisitos y Ejecución

Para ejecutar los generadores y visualizar los casos de prueba, asegúrese de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy scikit-learn
```
Para probar un generador específico de la Fase 1, ejecute:

```bash
python myquestions/question-0001-usecase-generator.py
```
Para validar una solución de la Fase 2, se recomienda ejecutar el generador de caso de uso correspondiente y comparar la salida esperada con la función implementada en `myanswers/answer-XXXX.py`.

Ejemplo general:

```python
from myanswers.answer_XXXX import nombre_de_la_funcion

entrada, salida_esperada = generar_caso_de_uso()
salida_real = nombre_de_la_funcion(**entrada)
```

---
## 🧪 Validación de la Fase 2

Las soluciones fueron desarrolladas siguiendo la lógica de los generadores de casos de uso asignados, respetando:

- El nombre exacto de cada función.
- El nombre exacto de cada archivo `answer-XXXX.py`.
- El tipo de dato esperado por el validador.
- El orden de columnas, listas, matrices o DataFrames cuando aplica.
- Los parámetros específicos usados en los modelos o transformaciones.

En caso de que una solución no coincida exactamente con el resultado esperado por el generador, la justificación correspondiente debe registrarse en `myanswers/diagnostico.txt`.

---

## 📝 Nota de Desarrollo

Todos los scripts han sido diseñados bajo principios de claridad, reproducibilidad y compatibilidad con el proceso de validación automática del curso. En la Fase 2 se priorizó replicar de forma precisa la lógica esperada por cada generador para garantizar que las soluciones sean verificables por el validador.