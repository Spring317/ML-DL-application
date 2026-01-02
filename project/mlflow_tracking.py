import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MLflowTracker:
    def __init__(self, experiment_name="BBC_Text_Classification", tracking_uri="./mlruns"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (local path or remote server)
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def log_preprocessing_params(self, params):
        """Log preprocessing parameters"""
        with mlflow.start_run(nested=True, run_name="preprocessing"):
            for key, value in params.items():
                mlflow.log_param(key, value)
    
    def log_sklearn_model(self, model, model_name, X_train, X_test, y_train, y_test, 
                          preprocessing_params=None, model_params=None):
        """
        Log scikit-learn model with MLflow
        
        Args:
            model: Trained sklearn model
            model_name: Name of the model
            X_train, X_test, y_train, y_test: Training and test data
            preprocessing_params: Dictionary of preprocessing parameters
            model_params: Dictionary of model parameters
        """
        with mlflow.start_run(run_name=model_name):
            # Log preprocessing parameters
            if preprocessing_params:
                for key, value in preprocessing_params.items():
                    mlflow.log_param(f"preprocessing_{key}", value)
            
            # Log model parameters
            if model_params:
                for key, value in model_params.items():
                    mlflow.log_param(key, value)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(y_train))
            mlflow.log_metric("test_samples", len(y_test))
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)
            
            # Log confusion matrix as artifact
            self._log_confusion_matrix(y_test, y_pred, model_name)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return mlflow.active_run().info.run_id
    
    def log_keras_model(self, model, model_name, history, X_train, X_test, y_train, y_test,
                       preprocessing_params=None, model_params=None):
        """
        Log Keras model with MLflow
        
        Args:
            model: Trained Keras model
            model_name: Name of the model
            history: Training history object
            X_train, X_test, y_train, y_test: Training and test data
            preprocessing_params: Dictionary of preprocessing parameters
            model_params: Dictionary of model parameters
        """
        with mlflow.start_run(run_name=model_name):
            # Log preprocessing parameters
            if preprocessing_params:
                for key, value in preprocessing_params.items():
                    mlflow.log_param(f"preprocessing_{key}", value)
            
            # Log model parameters
            if model_params:
                for key, value in model_params.items():
                    mlflow.log_param(key, value)
            
            # Log training history
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'],
                history.history['acc'],
                history.history['val_loss'],
                history.history['val_acc']
            )):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Evaluate model
            results = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metric("test_loss", results[0])
            mlflow.log_metric("test_accuracy", results[1])
            mlflow.log_metric("train_samples", len(y_train))
            mlflow.log_metric("test_samples", len(y_test))
            
            # Log training plots
            self._log_training_plots(history, model_name)
            
            # Log model
            mlflow.keras.log_model(model, "model")
            
            return mlflow.active_run().info.run_id
    
    def _log_confusion_matrix(self, y_true, y_pred, model_name):
        """Create and log confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Clean up
        if os.path.exists(cm_path):
            os.remove(cm_path)
    
    def _log_training_plots(self, history, model_name):
        """Create and log training plots"""
        # Accuracy plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['acc'], 'go-', label='Train accuracy')
        plt.plot(history.history['val_acc'], 'g-', label='Validate accuracy')
        plt.title('Train and validate accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], 'ro-', label='Train loss')
        plt.plot(history.history['val_loss'], 'r-', label='Validate loss')
        plt.title('Train and validate loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        # Save and log
        plot_path = f"training_plots_{model_name}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Clean up
        if os.path.exists(plot_path):
            os.remove(plot_path)
    
    def compare_runs(self, metric="accuracy"):
        """Compare all runs in the experiment"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if len(runs) == 0:
            print("No runs found in this experiment")
            return None
        
        # Sort by metric
        runs_sorted = runs.sort_values(f"metrics.{metric}", ascending=False)
        
        print(f"\nTop 10 runs sorted by {metric}:")
        print(runs_sorted[['run_id', 'tags.mlflow.runName', f'metrics.{metric}']].head(10))
        
        return runs_sorted
    
    def get_best_model(self, metric="accuracy"):
        """Get the best model based on a metric"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if len(runs) == 0:
            print("No runs found in this experiment")
            return None
        
        best_run = runs.loc[runs[f'metrics.{metric}'].idxmax()]
        
        print(f"\nBest model (by {metric}):")
        print(f"Run ID: {best_run['run_id']}")
        print(f"Model: {best_run['tags.mlflow.runName']}")
        print(f"{metric.capitalize()}: {best_run[f'metrics.{metric}']}")
        
        return best_run