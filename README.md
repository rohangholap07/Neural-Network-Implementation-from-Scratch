# Neural-Network-Implementation-from-Scratch

Neural Network Implementation from Scratch
Course: Generative AI Lab
Department: CSE AIML
Class: T.Y. Tech

Student Information
Name of Student: Rohan Gorakh Gholap
PRN Number: 202401110004
Batch: A1
Date of Submission: 16 August
Objective
Implement a simple feedforward neural network from scratch in Python without using an in-built deep learning library. The implementation demonstrates the forward pass, backpropagation, loss calculation, and training using gradient descent.

The neural-network mathematics and training loop are implemented manually using NumPy. Scikit-learn is used only for the Iris dataset, train/test split, and feature scaling.

1. Problem Definition
Dataset
The Iris dataset contains 150 flower samples from three classes:

Setosa
Versicolor
Virginica
Each sample has four numerical features:

Sepal length
Sepal width
Petal length
Petal width
Task
This is a multi-class classification problem. The neural network predicts which of the three Iris species a sample belongs to.

2. Methodology
Neural Network Architecture
Input layer: 4 neurons
Hidden layer: 8 neurons
Output layer: 3 neurons
Hidden activation: ReLU
Output activation: Softmax
Loss function: Categorical Cross-Entropy
Optimizer: Batch Gradient Descent
Forward Pass
For each layer, the weighted sum is calculated and an activation function is applied.

Backpropagation
The gradients of the loss with respect to the weights and biases are calculated using the chain rule. The parameters are then updated using gradient descent.


# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("Libraries imported successfully.")

     

# Load the Iris dataset
iris = load_iris()

X = iris.data.astype(np.float64)
y = iris.target.astype(int)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Class names:", iris.target_names)

     

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize features using training-set statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode target labels for categorical cross-entropy
num_classes = 3
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Input features:", X_train.shape[1])

     
3. Neural Network from Scratch
The following cells implement:

ReLU activation and derivative
Softmax activation
Categorical cross-entropy loss
Forward propagation
Backpropagation
Gradient descent
No TensorFlow, PyTorch, Keras, or other deep-learning framework is used.


# Activation functions

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):

    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    # Small epsilon prevents log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

     

# Initialize neural-network parameters

input_size = 4
hidden_size = 8
output_size = 3


W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))


W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
b2 = np.zeros((1, output_size))

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

     

# Forward pass

def forward_pass(X):

    Z1 = X @ W1 + b1
    A1 = relu(Z1)


    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# Test one forward pass
Z1, A1, Z2, A2 = forward_pass(X_train)

print("Hidden activation shape:", A1.shape)
print("Output probability shape:", A2.shape)
print("First sample probabilities:", A2[0])

     
4. Backpropagation and Gradient Descent
For softmax followed by categorical cross-entropy:

dZ2 = A2 - Y

Then:

dW2 = A1.T @ dZ2 / m
db2 = sum(dZ2) / m
dA1 = dZ2 @ W2.T
dZ1 = dA1 * ReLU'(Z1)
dW1 = X.T @ dZ1 / m
db1 = sum(dZ1) / m
Finally, gradient descent updates every parameter in the opposite direction of its gradient.


# Train the network using batch gradient descent

learning_rate = 0.05
epochs = 1000

loss_history = []
accuracy_history = []

m = X_train.shape[0]

for epoch in range(epochs):

    Z1, A1, Z2, A2 = forward_pass(X_train)

    loss = categorical_cross_entropy(y_train_one_hot, A2)


    dZ2 = (A2 - y_train_one_hot) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Track training performance
    predictions = np.argmax(A2, axis=1)
    train_accuracy = np.mean(predictions == y_train)

    loss_history.append(loss)
    accuracy_history.append(train_accuracy)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:4d}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Training Accuracy: {train_accuracy * 100:.2f}%"
        )

     

# Plot training loss

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.show()

     

# Plot training accuracy

plt.figure(figsize=(8, 5))
plt.plot(np.array(accuracy_history) * 100)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy vs Epoch")
plt.grid(True)
plt.show()

     
5. Model Evaluation
The trained model is evaluated on the unseen test set. Accuracy and a confusion matrix are calculated manually so that the evaluation does not depend on a deep-learning library.


# Evaluate on test data

_, _, _, test_probabilities = forward_pass(X_test)
y_pred = np.argmax(test_probabilities, axis=1)

test_accuracy = np.mean(y_pred == y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

     

# Build a confusion matrix manually

confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for actual, predicted in zip(y_test, y_pred):
    confusion_matrix[actual, predicted] += 1

print("Confusion Matrix:")
print(confusion_matrix)

     

# Display confusion matrix

plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix)
plt.colorbar()

plt.xticks(range(num_classes), iris.target_names)
plt.yticks(range(num_classes), iris.target_names)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, confusion_matrix[i, j],
                 ha="center", va="center")

plt.show()

     

# Show a few predictions

print("Sample Predictions")
print("-" * 60)

for i in range(min(10, len(X_test))):
    actual_name = iris.target_names[y_test[i]]
    predicted_name = iris.target_names[y_pred[i]]
    confidence = test_probabilities[i, y_pred[i]] * 100

    print(
        f"Sample {i+1:2d}: "
        f"Actual = {actual_name:10s} | "
        f"Predicted = {predicted_name:10s} | "
        f"Confidence = {confidence:.2f}%"
    )

     
6. Result and Conclusion
The feedforward neural network was successfully implemented from scratch using NumPy. The implementation includes a ReLU hidden layer, Softmax output layer, categorical cross-entropy loss, backpropagation, and batch gradient descent.

The model is trained on the Iris dataset and evaluated on unseen test samples. The loss and accuracy plots show the learning behavior during training, while the confusion matrix shows the classification performance for each Iris class.

Academic Integrity Declaration
I, [Your Name], confirm that the work submitted in this assignment is my own and has been completed following academic integrity guidelines.

GitHub Repository Link: https://github.com/rohangholap07/Neural-Network-Implementation-from-Scratch/new/main?readme=1

Signature: __________________________

Submission Checklist
Code file (Python Notebook)
Dataset / dataset loader
Visualizations
Model performance metrics
README file
GitHub repository link
