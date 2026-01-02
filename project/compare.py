import mlflow
from mlflow_tracking import MLflowTracker
import pandas as pd

# Initialize tracker
tracker = MLflowTracker(experiment_name="BBC_Text_Classification_All")

# Get all experiments
ml_experiment = mlflow.get_experiment_by_name("BBC_Text_ML_Models")
dl_experiment = mlflow.get_experiment_by_name("BBC_Text_DL_Models")

# Compare all models
print("=== ML Models ===")
ml_runs = mlflow.search_runs(experiment_ids=[ml_experiment.experiment_id])
print(ml_runs[['tags.mlflow.runName', 'metrics.accuracy']].sort_values('metrics.accuracy', ascending=False))

print("\n=== DL Models ===")
dl_runs = mlflow.search_runs(experiment_ids=[dl_experiment.experiment_id])
print(dl_runs[['tags.mlflow.runName', 'metrics.test_accuracy']].sort_values('metrics.test_accuracy', ascending=False))

# Overall best model
all_runs = pd.concat([
    ml_runs[['run_id', 'tags.mlflow.runName', 'metrics.accuracy']].rename(columns={'metrics.accuracy': 'accuracy'}),
    dl_runs[['run_id', 'tags.mlflow.runName', 'metrics.test_accuracy']].rename(columns={'metrics.test_accuracy': 'accuracy'})
])

print("\n=== Overall Best Model ===")
best_model = all_runs.loc[all_runs['accuracy'].idxmax()]
print(f"Model: {best_model['tags.mlflow.runName']}")
print(f"Accuracy: {best_model['accuracy']}")
print(f"Run ID: {best_model['run_id']}")