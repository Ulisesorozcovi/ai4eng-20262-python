# Modelos y simulacion 1 / 2026-2 - Fase 1: Generación de Casos de Uso

Este repositorio contiene la entrega de la **Fase 1** para el curso de Inteligencia Artificial aplicada a la Ingeniería. El objetivo principal es el diseño de problemas técnicos y la implementación de generadores de datos sintéticos para validar algoritmos de preprocesamiento.

## 👤 Información del Estudiante
* **Nombre:** Ulises Orozco Villegas
* **Correo Institucional:** Ulises.orozco@udea.edu.co
* **Institución:** Universidad de Antioquia

---

## 📂 Estructura del Repositorio

El proyecto se organiza siguiendo una jerarquía estricta para separar la lógica de las preguntas de las futuras respuestas:

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
├── myanswers/                  # Se desarrollara en la Fase 2
└── README.md                   # Documentación principal

---
## 🛠️ Descripción de los Módulos

1. Generación de Datos (Fase 1)
Cada archivo question-XXXX.txt plantea un escenario de ingeniería donde los datos requieren tratamiento. Los scripts de Python asociados (*-generator.py) implementan la lógica necesaria para Simulación de Errores, Generación Sintétic y Cálculo de Ground Truth.

2. Técnicas de Preprocesamiento Implementadas

En esta fase se han cubierto las siguientes técnicas críticas:
- Imputación: Uso de media y mediana mediante SimpleImputer.
- Escalado: Implementación de StandardScaler, MinMaxScaler y RobustScaler (especialmente útil para datos con outliers).

---
## 🚀 Requisitos y Ejecución

Para ejecutar los generadores y visualizar los casos de prueba, asegúrese de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy scikit-learn

Para probar un generador específico, ejecute:

```bash
python myquestions/question-0001-usecase-generator.py

Nota de Desarrollo: Todos los scripts han sido diseñados bajo principios de programación funcional y tipado estático (typing) para garantizar la robustez del código.