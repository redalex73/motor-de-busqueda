import ray
from utils.data_loader import load_data_paths
from ray_tasks.embedding_worker import compute_image_embedding, compute_text_embedding
from tracking.mlflow_helper import start_run, log_metrics, end_run
from search_engine import search
from eval import evaluate

def main():
    ray.init(include_dashboard=False)

    image_paths, text_paths = load_data_paths("data/")

    # Distribuir computación con Ray
    image_futures = [compute_image_embedding.remote(p) for p in image_paths]
    text_futures = [compute_text_embedding.remote(p) for p in text_paths]

    image_results = ray.get(image_futures)
    text_results = ray.get(text_futures)

    # Convertir a diccionario
    image_embeddings = {name: emb for name, emb in image_results}
    text_embeddings = {name: emb for name, emb in text_results}

    # Búsqueda semántica
    results = search(image_embeddings, text_embeddings)

    # Evaluación
    metrics = evaluate(results)

    # MLflow
    start_run("OpenCLIP_Exp_Distributed")
    log_metrics(metrics)
    end_run()

if __name__ == "__main__":
    main()

