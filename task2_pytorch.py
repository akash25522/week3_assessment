import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features without data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Input features:", X_train.shape[1])

# Custom Dataset
class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Create datasets
train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Test DataLoader
features, labels = next(iter(train_loader))

print("Batch features shape:", features.shape)
print("Batch labels shape:", labels.shape)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# Create model
model = MLP(input_size=X_train.shape[1])

# Count parameters
params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", params)
print(model)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 40
patience = 5

best_val_accuracy = 0
best_state = None
wait = 0

train_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping triggered.")
        break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

print(f"Best validation accuracy: {best_val_accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "best_model.pth")

# Evaluate original model
model.eval()

with torch.no_grad():
    original_outputs = model(X_val)
    original_predictions = (
        torch.sigmoid(original_outputs) >= 0.5
    ).float()

    original_accuracy = (
        original_predictions == y_val
    ).float().mean().item()

print("Original model accuracy:", round(original_accuracy, 4))


# Create a new model
loaded_model = MLP(input_size=X_train.shape[1])

# Load saved weights
loaded_model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)

loaded_model.eval()

# Evaluate reloaded model
with torch.no_grad():
    loaded_outputs = loaded_model(X_val)

    loaded_predictions = (
        torch.sigmoid(loaded_outputs) >= 0.5
    ).float()

    loaded_accuracy = (
        loaded_predictions == y_val
    ).float().mean().item()

print("Reloaded model accuracy:", round(loaded_accuracy, 4))

print(
    "Accuracy identical:",
    abs(original_accuracy - loaded_accuracy) < 1e-8
)

# Save model
torch.save(model.state_dict(), "best_model.pth")

# Evaluate original model
model.eval()

with torch.no_grad():
    original_outputs = model(X_val)
    original_predictions = (
        torch.sigmoid(original_outputs) >= 0.5
    ).float()

    original_accuracy = (
        original_predictions == y_val
    ).float().mean().item()

print("Original model accuracy:", round(original_accuracy, 4))


# Create a new model
loaded_model = MLP(input_size=X_train.shape[1])

# Load saved weights
loaded_model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)

loaded_model.eval()

# Evaluate reloaded model
with torch.no_grad():
    loaded_outputs = loaded_model(X_val)

    loaded_predictions = (
        torch.sigmoid(loaded_outputs) >= 0.5
    ).float()

    loaded_accuracy = (
        loaded_predictions == y_val
    ).float().mean().item()

print("Reloaded model accuracy:", round(loaded_accuracy, 4))

print(
    "Accuracy identical:",
    abs(original_accuracy - loaded_accuracy) < 1e-8
)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train.numpy(), y_train.numpy())

# Predict validation data
lr_predictions = lr_model.predict(X_val.numpy())

# Calculate accuracy
lr_accuracy = accuracy_score(
    y_val.numpy(),
    lr_predictions
)

print("\n--- Model Comparison ---")
print(f"PyTorch MLP Accuracy: {original_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

