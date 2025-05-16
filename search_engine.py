import torch
import numpy as np

def cosine_sim(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def search(image_embeddings: dict, text_embeddings: dict, top_k: int = 5) -> dict:
    """
    Para cada texto, devuelve las imágenes más similares.
    """
    results = {}
    for text_name, text_emb in text_embeddings.items():
        similarities = []
        for image_name, image_emb in image_embeddings.items():
            sim = cosine_sim(text_emb, image_emb)
            similarities.append((image_name, float(sim)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results[text_name] = similarities[:top_k]
    return results
