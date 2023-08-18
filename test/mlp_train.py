import numpy as np

# Define the dataset
data = [
    [0, 0, 1],
    [1, 2, -1],
    [2, -1, -1],
    [-2, 1, -1]
]

# Convert data to numpy arrays
X = np.array([row[:2] for row in data])
y = np.array([row[2] for row in data])

# Initialize parameters
learning_rate = 1
w = np.array([1, 0])
b = 1.5

# Perceptron training algorithm
for _ in range(50):  # Typically, you'd run for a certain number of epochs or until convergence
    for i in range(X.shape[0]):
        y_pred = np.sign(np.dot(w, X[i]) + b)

        # Update rule
        if y_pred != y[i]:
            w = w + learning_rate * (y[i] - y_pred) * X[i]
            b = b + learning_rate * (y[i] - y_pred)

print("Final weights:", w)
print("Final bias:", b)
