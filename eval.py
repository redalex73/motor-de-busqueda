def evaluate(results):
    # Evaluar la precisión de recuperación
    # Esta es una función de ejemplo; debes ajustarla según tus datos
    precision_at_k = {}
    for query, retrieved in results.items():
        # Suponiendo que el nombre del archivo de texto coincide con el de la imagen relevante
        relevant = query.replace('.txt', '.jpg')
        retrieved_names = [name for name, _ in retrieved]
        precision = int(relevant in retrieved_names) / len(retrieved_names)
        precision_at_k[query] = precision
    # Calcular la precisión promedio
    avg_precision = sum(precision_at_k.values()) / len(precision_at_k)
    return {"average_precision": avg_precision}
