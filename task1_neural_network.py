import numpy as np
import matplotlib.pyplot as plt
import torch


# 1. ReLU activation
def relu(x):
    return np.maximum(0, x)


# 2. Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# 3. Softmax activation
def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# 4. Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 5. Binary Cross-Entropy
def cross_entropy(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# Test activation functions
print("ReLU:", relu(np.array([-2, 0, 3])))

print("Sigmoid:", sigmoid(np.array([0, 2])))

print("Softmax:", np.round(
    softmax([2.0, 1.0, 0.1]), 3
))

print("MSE:", mse(
    np.array([1, 0]),
    np.array([0.8, 0.2])
))

print("Cross Entropy:", cross_entropy(
    np.array([1, 0]),
    np.array([0.8, 0.2])
))
# Step 7: PyTorch Autograd

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(2.0)
t = torch.tensor(10.0)

# Forward pass
prediction = w * x + b

# Calculate loss
loss = (prediction - t) ** 2

# Backpropagation
loss.backward()

print("\n--- PyTorch Autograd ---")
print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("w.grad:", w.grad.item())
print("b.grad:", b.grad.item())

# Step 10: Train a 2-4-1 Neural Network on XOR

np.random.seed(42)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# Initialize weights and biases
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))


# Training settings
learning_rate = 0.5
epochs = 5000

loss_history = []


# Training loop
for epoch in range(epochs):

    # 1. Forward pass
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # 2. Calculate binary cross-entropy loss
    loss = cross_entropy(y, y_pred)
    loss_history.append(loss)

    # 3. Backpropagation
    m = len(X)

    dz2 = (y_pred - y) / m

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # 4. Gradient descent updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# Final predictions
z1 = X @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
predictions = sigmoid(z2)

print("\n--- XOR Neural Network Results ---")
print("Final loss:", loss_history[-1])

print(
    "XOR predictions:",
    np.round(predictions.flatten(), 2)
)

print(
    "Predicted classes:",
    (predictions.flatten() >= 0.5).astype(int)
)


# Plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("XOR Neural Network Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("xor_loss.png")
plt.show()

# Step 12: Compare Learning Rates

def train_xor(lr, epochs=5000):
    np.random.seed(42)

    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        pred = sigmoid(z2)

        loss = cross_entropy(y, pred)
        losses.append(loss)

        m = len(X)
        dz2 = (pred - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    return losses


plt.figure(figsize=(8, 5))

for lr in [0.01, 0.5, 5]:
    losses = train_xor(lr)
    plt.plot(losses, label=f"Learning Rate = {lr}")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_rate_comparison.png")
plt.show()

# Dropout:
# Randomly disables some neurons during training to reduce overfitting.

# Batch Normalization:
# Normalizes layer activations to make training more stable.

# Early Stopping:
# Stops training when validation performance stops improving.