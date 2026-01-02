import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dropout, Flatten, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from mlflow_tracking import MLflowTracker

# Initialize MLflow tracker
tracker = MLflowTracker(experiment_name="BBC_Text_DL_Models")

# Download NLTK data
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("./data/bbc-text.csv", engine='python', encoding='UTF-8')

# Preprocessing parameters
vocabulary_size = 15000
max_text_len = 768
stemmer = SnowballStemmer('english')
stop_words = [word for word in stopwords.words('english') if word not in ["my","haven't","aren't","can","no", "why", "through", "herself", "she", "he", "himself", "you", "you're", "myself", "not", "here", "some", "do", "does", "did", "will", "don't", "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't", "mustn't", "shouldn't", "hadn't", "wasn't", "wouldn't"]]

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if not word in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Tokenization
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['cleaned_text'].values)

sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)
X_DeepLearning = pad_sequences(sequences, maxlen=max_text_len)

# Encode labels
df.loc[df['category'] == 'sport', 'LABEL'] = 0
df.loc[df['category'] == 'business', 'LABEL'] = 1
df.loc[df['category'] == 'politics', 'LABEL'] = 2
df.loc[df['category'] == 'tech', 'LABEL'] = 3
df.loc[df['category'] == 'entertainment', 'LABEL'] = 4

labels = to_categorical(df['LABEL'], num_classes=5)

# Split data
XX_train, XX_test, y_train, y_test = train_test_split(X_DeepLearning, labels, test_size=0.25, random_state=42)

print(f"Data shapes: {XX_train.shape}, {y_train.shape}, {XX_test.shape}, {y_test.shape}")

# Define preprocessing parameters for logging
preprocessing_params = {
    "vocabulary_size": vocabulary_size,
    "max_text_len": max_text_len,
    "stemming": True,
    "stopwords_removed": True,
    "padding": "post"
}

# Model parameters
epochs = 25
emb_dim = 256
batch_size = 50

model_params = {
    "epochs": epochs,
    "embedding_dim": emb_dim,
    "batch_size": batch_size,
    "lstm_units": 300,
    "dropout": 0.5,
    "recurrent_dropout": 0.5,
    "spatial_dropout": 0.8,
    "optimizer": "Adam",
    "loss": "categorical_crossentropy"
}

# Build LSTM model
model_lstm1 = Sequential()
model_lstm1.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning.shape[1]))
model_lstm1.add(SpatialDropout1D(0.8))
model_lstm1.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Flatten())
model_lstm1.add(Dense(64, activation='relu'))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Dense(5, activation='softmax'))
model_lstm1.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

print(model_lstm1.summary())

# Callbacks
checkpoint_callback = ModelCheckpoint(filepath="lstm-1-layer-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)
early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks = [checkpoint_callback, early_stopping_callback, reduce_lr_callback]

# Train model
history_lstm1 = model_lstm1.fit(
    XX_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(XX_test, y_test),
    callbacks=callbacks
)

# Log model with MLflow
print("\n=== Logging LSTM Model to MLflow ===")
tracker.log_keras_model(
    model=model_lstm1,
    model_name="Bidirectional_LSTM",
    history=history_lstm1,
    X_train=XX_train,
    X_test=XX_test,
    y_train=y_train,
    y_test=y_test,
    preprocessing_params=preprocessing_params,
    model_params=model_params
)

# Compare all runs
print("\n=== Comparing All Models ===")
tracker.compare_runs(metric="test_accuracy")

# Get best model
print("\n=== Best Model ===")
tracker.get_best_model(metric="test_accuracy")
