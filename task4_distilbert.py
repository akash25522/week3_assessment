import os
import time
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "saved_distilbert"
CSV_FILE = "support_tickets.csv"

# Load dataset
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]].rename(
        columns={"label_id": "labels"}
    ).reset_index(drop=True)
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]].rename(
        columns={"label_id": "labels"}
    ).reset_index(drop=True)
)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

columns = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

# Load DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="distilbert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    save_total_limit=1
)


def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune
start_train = time.time()
trainer.train()
training_time = time.time() - start_train

# Evaluate
metrics = trainer.evaluate()
print("\nDistilBERT Evaluation:")
print(metrics)

# Predict
start_inference = time.time()
predictions = trainer.predict(test_dataset)
inference_time = time.time() - start_inference

predicted_ids = np.argmax(predictions.predictions, axis=-1)
true_ids = predictions.label_ids

print("\nClassification Report:")
print(
    classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    )
)

# Save model and tokenizer
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("\nModel saved:", MODEL_DIR)

# Save metrics
distilbert_accuracy = accuracy_score(true_ids, predicted_ids)
distilbert_f1 = f1_score(
    true_ids,
    predicted_ids,
    average="macro",
    zero_division=0
)

results = pd.DataFrame([{
    "Model": "DistilBERT",
    "Accuracy": distilbert_accuracy,
    "Macro F1": distilbert_f1,
    "Training Time (seconds)": training_time,
    "Inference Time (seconds)": inference_time
}])

results.to_csv("distilbert_metrics.csv", index=False)
print("\nDistilBERT metrics saved to distilbert_metrics.csv")

# Test saved model using pipeline
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=0 if torch.cuda.is_available() else -1
)

print("\nPipeline prediction:")
print(classifier("I cannot log in to my account"))