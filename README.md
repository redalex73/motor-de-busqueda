# Proyecto de Búsqueda Semántica Distribuida con Ray y OpenCLIP

Este proyecto implementa un sistema básico de búsqueda semántica distribuida utilizando Ray para paralelizar el cálculo de embeddings de imágenes y textos, junto con OpenCLIP para la extracción de características semánticas. Está diseñado para demostrar cómo combinar procesamiento distribuido, búsqueda vectorial y evaluación de resultados, integrando además MLflow para el seguimiento y análisis de métricas.

## Características principales

- Cálculo paralelo de embeddings de imágenes y textos con Ray.
- Búsqueda semántica basada en similitud de coseno entre embeddings.
- Evaluación de la precisión de recuperación mediante métricas simples.
- Registro y seguimiento de experimentos con MLflow.
- Código modular y fácilmente ampliable.

## Requisitos

- Python 3.8+
- Ray (versión recomendada: 2.45.0)
- torch
- numpy
- mlflow
- OpenCLIP (instalación manual o a través de pip)
