import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from textblob import Word 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from mlflow_tracking import MLflowTracker

# Initialize MLflow tracker
tracker = MLflowTracker(experiment_name="BBC_Text_ML_Models")

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("./data/bbc-text.csv", engine='python', encoding='UTF-8')

# Data cleaning and preprocessing
df['text'] = df['text'].fillna("")
df['lower_case'] = df['text'].apply(lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))
df['alphabatic'] = df['lower_case'].apply(lambda x: re.sub(r'[^a-zA-Z\']', ' ', x)).apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
df['without-link'] = df['alphabatic'].apply(lambda x: re.sub(r'http\S+', '', x))

tokenizer = RegexpTokenizer(r'\w+')
df['Special_word'] = df.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)

stop = [word for word in stopwords.words('english') if word not in ["my","haven't","aren't","can","no", "why", "through", "herself", "she", "he", "himself", "you", "you're", "myself", "not", "here", "some", "do", "does", "did", "will", "don't", "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't", "mustn't", "shouldn't", "hadn't", "wasn't", "wouldn't"]]

df['stop_words'] = df['Special_word'].apply(lambda x: [item for item in x if item not in stop])
df['stop_words'] = df['stop_words'].astype('str')

df['short_word'] = df['stop_words'].str.findall('\w{2,}')
df['string'] = df['short_word'].str.join(' ') 

# Lemmatization
df['Text'] = df['string'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Define preprocessing parameters
preprocessing_params = {
    "min_word_length": 2,
    "stopwords_removed": True,
    "lemmatization": True,
    "ngram_range": "(1, 2)",
    "tfidf_norm": "l2",
    "tfidf_sublinear": True
}

# Split data
x_train, x_test, y_train, y_test = train_test_split(df["Text"], df["category"], test_size=0.25, random_state=42)

# Vectorization
count_vect = CountVectorizer(ngram_range=(1, 2))
transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

print(f"Data shapes: {x_train_tfidf.shape}, {x_test_tfidf.shape}, {y_train.shape}, {y_test.shape}")

# Train and log Logistic Regression
print("\n=== Training Logistic Regression ===")
lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
lr.fit(x_train_tfidf, y_train)

lr_params = {
    "C": 2,
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "auto"
}

tracker.log_sklearn_model(
    model=lr,
    model_name="Logistic_Regression",
    X_train=x_train_tfidf,
    X_test=x_test_tfidf,
    y_train=y_train,
    y_test=y_test,
    preprocessing_params=preprocessing_params,
    model_params=lr_params
)

# Train and log Naive Bayes
print("\n=== Training Naive Bayes ===")
mnb = MultinomialNB()
mnb.fit(x_train_tfidf, y_train)

mnb_params = {
    "alpha": 1.0,
    "fit_prior": True
}

tracker.log_sklearn_model(
    model=mnb,
    model_name="Naive_Bayes",
    X_train=x_train_tfidf,
    X_test=x_test_tfidf,
    y_train=y_train,
    y_test=y_test,
    preprocessing_params=preprocessing_params,
    model_params=mnb_params
)

# Compare all runs
print("\n=== Comparing All Models ===")
tracker.compare_runs(metric="accuracy")

# Get best model
print("\n=== Best Model ===")
tracker.get_best_model(metric="accuracy")

