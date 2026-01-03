import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ktrain
from ktrain import text
import mlflow
import mlflow.keras
from mlflow_tracking import MLflowTracker
import os
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Initialize MLflow tracker
tracker = MLflowTracker(experiment_name="BBC_Text_BERT_Models")

print("=" * 80)
print("DistilBERT Fine-tuning for BBC Text Classification using ktrain")
print("=" * 80)

# Load dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv("./data/bbc-text.csv", engine='python', encoding='UTF-8')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Categories: {df['category'].unique()}")
print(f"Category distribution:\n{df['category'].value_counts()}")

# Prepare data
print("\n[2/7] Preparing data...")
X = df['text'].values
y = df['category'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Define preprocessing parameters
preprocessing_params = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "train_test_split": 0.25,
    "random_state": 42,
    "stratified": True
}

# Define model parameters
model_name = 'distilbert-base-uncased'
max_length = 512
batch_size = 8
learning_rate = 2e-5
epochs = 15

model_params = {
    "model_name": model_name,
    "max_length": max_length,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "optimizer": "adam",
    "freeze_layers": False
}

# Start MLflow run
print("\n[3/7] Starting MLflow tracking...")
with mlflow.start_run(run_name="DistilBERT_ktrain") as run:
    
    # Log parameters
    print("Logging parameters to MLflow...")
    for key, value in preprocessing_params.items():
        mlflow.log_param(f"preprocessing_{key}", value)
    
    for key, value in model_params.items():
        mlflow.log_param(key, value)
    
    # Log dataset metrics
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    mlflow.log_metric("num_classes", len(np.unique(y)))
    
    # Build the transformer model
    print("\n[4/7] Building DistilBERT model with ktrain...")
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_length}")
    print("This may take a few minutes to download the model if it's the first time...")
    
    # Create text preprocessor
    t = text.Transformer(
        model_name,
        maxlen=max_length,
        class_names=list(np.unique(y))
    )
    
    # Preprocess training and test data
    print("\n[5/7] Preprocessing text data...")
    train_dataset = t.preprocess_train(X_train, y_train)
    test_dataset = t.preprocess_test(X_test, y_test)
    
    # Build and compile model
    print("Building model architecture...")
    model = t.get_classifier()
    
    # Create learner
    learner = ktrain.get_learner(
        model,
        train_data=train_dataset,
        val_data=test_dataset,
        batch_size=batch_size
    )
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {model.count_params():,}")
    
    # Train the model
    print(f"\n[6/7] Training DistilBERT model for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 80)
    
    # Train with learning rate schedule
    history = learner.fit_onecycle(
        learning_rate,
        epochs,
        checkpoint_folder='./bert_checkpoints'
    )
    
    print("\nTraining completed!")
    
    # Log training metrics
    print("\nLogging training metrics...")
    if hasattr(learner, 'history') and learner.history:
        hist = learner.history.history
        for epoch in range(len(hist.get('loss', []))):
            if 'loss' in hist:
                mlflow.log_metric("train_loss", hist['loss'][epoch], step=epoch)
            if 'accuracy' in hist:
                mlflow.log_metric("train_accuracy", hist['accuracy'][epoch], step=epoch)
            if 'val_loss' in hist:
                mlflow.log_metric("val_loss", hist['val_loss'][epoch], step=epoch)
            if 'val_accuracy' in hist:
                mlflow.log_metric("val_accuracy", hist['val_accuracy'][epoch], step=epoch)
    
    # Evaluate the model
    print("\n[7/7] Evaluating model on test set...")
    
    # Get predictions
    print("Making predictions...")
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    y_pred = predictor.predict(X_test)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Log test accuracy
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Classification report
    print("\nClassification Report:")
    print("=" * 80)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Log classification metrics
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=np.unique(y),
        yticklabels=np.unique(y)
    )
    plt.title('DistilBERT - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save and log confusion matrix
    cm_path = "bert_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # Clean up
    if os.path.exists(cm_path):
        os.remove(cm_path)
    
    # Save the predictor
    print("\nSaving DistilBERT predictor...")
    predictor.save('./bert_predictor')
    
    # Log the model to MLflow
    print("Logging model to MLflow...")
    mlflow.keras.log_model(learner.model, "model")
    
    # Log predictor as artifact
    mlflow.log_artifacts('./bert_predictor', artifact_path='predictor')
    
    print("\n" + "=" * 80)
    print("DistilBERT Fine-tuning Completed!")
    print("=" * 80)
    print(f"\nRun ID: {run.info.run_id}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nModel and artifacts saved to MLflow")
    print(f"Predictor saved to: ./bert_predictor")
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui --backend-store-uri ./mlruns --port 5000")
    print("=" * 80)

# Compare with other models
print("\n\nComparing with other models...")
print("=" * 80)
tracker.compare_runs(metric="test_accuracy")

# Get best model
print("\n" + "=" * 80)
tracker.get_best_model(metric="test_accuracy")
print("=" * 80)

# Example predictions
print("\n\nExample Predictions:")
print("=" * 80)
predictor = ktrain.load_predictor('./bert_predictor')

sample_texts = [
    "The stock market reached new heights today as tech companies reported strong earnings.",
    "The football team won the championship after a thrilling final match.",
    "Scientists discover new treatment for cancer in groundbreaking research.",
    "The government announced new policies to address climate change.",
    "The latest smartphone features advanced AI capabilities and improved camera."
]

for i, text in enumerate(sample_texts, 1):
    prediction = predictor.predict(text)
    print(f"\n{i}. Text: {text[:80]}...")
    print(f"   Predicted Category: {prediction}")

print("\n" + "=" * 80)
print("Script completed successfully!")
print("=" * 80)
