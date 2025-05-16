import mlflow

def start_run(run_name="default"):
    mlflow.start_run(run_name=run_name)

def log_metrics(metrics_dict):
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value)

def log_param_dict(params):
    for key, val in params.items():
        mlflow.log_param(key, val)

def end_run():
    mlflow.end_run()
