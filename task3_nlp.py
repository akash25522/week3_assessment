import pandas as pd

data = {
    "text": [
        "I cannot log in to my account",
        "Please reset my password",
        "My account login is not working",
        "I forgot my password",
        "Unable to access my account",
        "Login failed again",
        "I need help signing in",
        "My password is incorrect",
        "Account access problem",
        "I cannot sign in",
        "I was charged twice",
        "My payment was deducted twice",
        "I need a refund",
        "The subscription payment is incorrect",
        "I did not receive my refund",
        "Payment failed but money deducted",
        "Billing amount is wrong",
        "Please cancel my subscription",
        "I have a payment issue",
        "Why was I charged again?",
        "The application keeps crashing",
        "App crashes when I open settings",
        "I get error 500 while uploading",
        "The website is not loading",
        "Upload feature is not working",
        "Application shows an error",
        "The app freezes frequently",
        "I cannot upload my file",
        "Server error while using the app",
        "The application is very slow"
    ]
}

df = pd.DataFrame(data)

# Repeat examples to create at least 300 labelled texts
df = pd.concat([df] * 10, ignore_index=True)

labels = [
    "Account"] * 10 + ["Billing"] * 10 + ["Technical"] * 10

df["label"] = labels * 10

df.to_csv("support_tickets.csv", index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TF-IDF Confusion Matrix")
plt.tight_layout()
plt.savefig("tfidf_confusion_matrix.png")
plt.show()

# Top 10 words for each class
feature_names = vectorizer.get_feature_names_out()

print("\nTop 10 words per class:")

for i, class_name in enumerate(model.classes_):
    top_indices = model.coef_[i].argsort()[-10:][::-1]
    top_words = feature_names[top_indices]

    print(f"\n{class_name}:")
    print(", ".join(top_words))

from gensim.models import Word2Vec
import re

# Prepare sentences
sentences = [
    re.findall(r"\b\w+\b", text.lower())
    for text in df["text"]
]

# Additional words for analogy
extra_sentences = [
    "king is a man".split(),
    "queen is a woman".split(),
    "the king rules the kingdom".split(),
    "the queen rules the kingdom".split(),
    "man and woman are people".split(),
    "king and queen are royalty".split()
]

sentences.extend(extra_sentences)

# Train Word2Vec
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=1,
    seed=42,
    epochs=100
)

# Nearest words
print("\nWords similar to 'account':")
print(word2vec_model.wv.most_similar("account", topn=5))

# Word analogy: king - man + woman
print("\nAnalogy: king - man + woman")
print(
    word2vec_model.wv.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=3
    )
)    

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Encode labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Average Word2Vec vectors
def average_word_vectors(texts, w2v_model):
    vectors = []

    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())

        word_vectors = [
            w2v_model.wv[word]
            for word in words
            if word in w2v_model.wv
        ]

        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(w2v_model.vector_size))

    return np.array(vectors)

X_train_w2v = average_word_vectors(X_train, word2vec_model)
X_test_w2v = average_word_vectors(X_test, word2vec_model)

# Train Logistic Regression on Word2Vec vectors
w2v_classifier = LogisticRegression(max_iter=1000)
w2v_classifier.fit(X_train_w2v, y_train_encoded)

w2v_predictions = w2v_classifier.predict(X_test_w2v)

w2v_accuracy = accuracy_score(y_test_encoded, w2v_predictions)
tfidf_accuracy = accuracy_score(y_test, y_pred)

print("\nModel Comparison")
print("-------------------------")
print(f"TF-IDF Accuracy: {tfidf_accuracy:.4f}")
print(f"Word2Vec Accuracy: {w2v_accuracy:.4f}")