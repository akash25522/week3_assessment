import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformers import pipeline, AutoConfig

# Load dataset
df = pd.read_csv("support_tickets.csv")
df = df.dropna(subset=["text", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].astype(str),
    df["label"].astype(str),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF model
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

tfidf_model = LogisticRegression(max_iter=1000)

start = time.time()
tfidf_model.fit(X_train_tfidf, y_train)
tfidf_train_time = time.time() - start

start = time.time()
tfidf_predictions = tfidf_model.predict(X_test_tfidf)
tfidf_inference_time = time.time() - start

tfidf_accuracy = accuracy_score(y_test, tfidf_predictions)
tfidf_f1 = f1_score(
    y_test, tfidf_predictions, average="macro", zero_division=0
)

# DistilBERT model
config = AutoConfig.from_pretrained("saved_distilbert")
classifier = pipeline(
    "text-classification",
    model="saved_distilbert",
    tokenizer="saved_distilbert",
    device=-1
)

start = time.time()
bert_predictions = classifier(
    X_test.tolist(),
    batch_size=8,
    truncation=True
)
bert_inference_time = time.time() - start

id2label = config.id2label

bert_labels = []
for prediction in bert_predictions:
    label = prediction["label"]

    if label.startswith("LABEL_"):
        label_id = int(label.replace("LABEL_", ""))
        label = id2label[label_id]

    bert_labels.append(label)

bert_accuracy = accuracy_score(y_test, bert_labels)
bert_f1 = f1_score(
    y_test, bert_labels, average="macro", zero_division=0
)

# Load DistilBERT training time
try:
    bert_metrics = pd.read_csv("distilbert_metrics.csv").iloc[0]
    bert_train_time = bert_metrics["Training Time (seconds)"]
except (FileNotFoundError, KeyError):
    bert_train_time = "Not available"

# Comparison table
comparison = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": round(tfidf_accuracy, 4),
        "Macro F1": round(tfidf_f1, 4),
        "Training Time (seconds)": round(tfidf_train_time, 2),
        "Inference Time (seconds)": round(tfidf_inference_time, 4)
    },
    {
        "Model": "DistilBERT",
        "Accuracy": round(bert_accuracy, 4),
        "Macro F1": round(bert_f1, 4),
        "Training Time (seconds)": bert_train_time,
        "Inference Time (seconds)": round(bert_inference_time, 2)
    }
])

print("\nModel Comparison")
print(comparison.to_string(index=False))

comparison.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")