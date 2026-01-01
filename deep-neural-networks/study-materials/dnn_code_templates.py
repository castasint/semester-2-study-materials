# 🐍 DNN Python Code Templates
### Ready-to-Use Code for Exam | AIMLCZG511

---

## 1. Perceptron Implementation

```python
import numpy as np

def perceptron_train(X, y, eta=1.0, epochs=100):
    """
    Train a perceptron classifier.
    
    Parameters:
    -----------
    X : np.array, shape (n_samples, n_features)
    y : np.array, shape (n_samples,) - labels (+1 or -1)
    eta : float - learning rate
    epochs : int - number of training epochs
    
    Returns:
    --------
    w : np.array - learned weights (including bias as w[0])
    """
    # Add bias column (x₀ = 1)
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Initialize weights to zeros
    w = np.zeros(X_bias.shape[1])
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute weighted sum
            z = np.dot(w, X_bias[i])
            
            # Apply sign activation
            y_pred = 1 if z >= 0 else -1
            
            # Update if prediction is wrong
            if y_pred != y[i]:
                w = w + eta * (y[i] - y_pred) * X_bias[i]
    
    return w

def perceptron_predict(X, w):
    """Predict using trained perceptron."""
    X_bias = np.c_[np.ones(X.shape[0]), X]
    z = np.dot(X_bias, w)
    return np.where(z >= 0, 1, -1)

# Example: AND gate (binary 0/1 version)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([-1, -1, -1, 1])  # bipolar
w = perceptron_train(X_and, y_and)
print("AND gate weights:", w)
```

---

## 2. Linear Regression with Gradient Descent

```python
import numpy as np

def linear_regression_gd(X, y, learning_rate=0.01, epochs=1000):
    """
    Train linear regression using batch gradient descent.
    
    Parameters:
    -----------
    X : np.array, shape (n_samples, n_features)
    y : np.array, shape (n_samples,)
    learning_rate : float
    epochs : int
    
    Returns:
    --------
    w : np.array - weights (including bias)
    history : list - loss at each epoch
    """
    # Add bias column
    X_b = np.c_[np.ones(X.shape[0]), X]
    n_samples = X.shape[0]
    
    # Initialize weights
    w = np.zeros(X_b.shape[1])
    history = []
    
    for epoch in range(epochs):
        # Forward pass: predictions
        y_pred = X_b @ w
        
        # Compute error
        error = y_pred - y
        
        # Compute MSE loss
        loss = (1/(2*n_samples)) * np.sum(error**2)
        history.append(loss)
        
        # Compute gradient
        gradient = (1/n_samples) * X_b.T @ error
        
        # Update weights
        w = w - learning_rate * gradient
    
    return w, history

def predict(X, w):
    """Predict using trained linear model."""
    X_b = np.c_[np.ones(X.shape[0]), X]
    return X_b @ w

# Example usage
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 5])
w, loss_history = linear_regression_gd(X, y, learning_rate=0.1, epochs=100)
print("Weights:", w)
print("Final loss:", loss_history[-1])
```

---

## 3. Logistic Regression (Binary Classification)

```python
import numpy as np

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_pred, y_true):
    """Compute binary cross-entropy loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def logistic_regression_sgd(X, y, learning_rate=0.1, epochs=100):
    """
    Train logistic regression using SGD.
    
    Parameters:
    -----------
    X : np.array, shape (n_samples, n_features)
    y : np.array, shape (n_samples,) - labels (0 or 1)
    
    Returns:
    --------
    w : np.array - learned weights
    """
    X_b = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_b.shape[1])
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X))
        
        for i in indices:
            # Forward pass
            z = np.dot(w, X_b[i])
            y_pred = sigmoid(z)
            
            # Compute gradient
            gradient = (y_pred - y[i]) * X_b[i]
            
            # Update weights
            w = w - learning_rate * gradient
    
    return w

def logistic_predict(X, w, threshold=0.5):
    """Predict class labels."""
    X_b = np.c_[np.ones(X.shape[0]), X]
    probs = sigmoid(X_b @ w)
    return (probs >= threshold).astype(int)

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])
w = logistic_regression_sgd(X, y, learning_rate=0.5, epochs=100)
print("Weights:", w)
print("Predictions:", logistic_predict(X, w))
```

---

## 4. Softmax and Multi-class Classification

```python
import numpy as np

def softmax(z):
    """
    Compute softmax probabilities.
    
    Parameters:
    -----------
    z : np.array - logits, shape (n_classes,) or (n_samples, n_classes)
    
    Returns:
    --------
    np.array - probabilities that sum to 1
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def categorical_cross_entropy(y_pred, y_true):
    """
    Compute categorical cross-entropy loss.
    
    Parameters:
    -----------
    y_pred : np.array - predicted probabilities, shape (n_samples, n_classes)
    y_true : np.array - one-hot encoded labels, shape (n_samples, n_classes)
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def multiclass_train(X, y, n_classes, learning_rate=0.01, epochs=100):
    """
    Train multi-class classifier using mini-batch gradient descent.
    
    Parameters:
    -----------
    X : np.array, shape (n_samples, n_features)
    y : np.array, shape (n_samples,) - integer class labels
    n_classes : int
    
    Returns:
    --------
    W : np.array, shape (n_features+1, n_classes)
    """
    X_b = np.c_[np.ones(X.shape[0]), X]
    n_samples, n_features = X_b.shape
    
    # One-hot encode labels
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), y] = 1
    
    # Initialize weights
    W = np.zeros((n_features, n_classes))
    
    for epoch in range(epochs):
        # Forward pass
        z = X_b @ W
        y_pred = softmax(z)
        
        # Compute gradient
        gradient = (1/n_samples) * X_b.T @ (y_pred - Y)
        
        # Update weights
        W = W - learning_rate * gradient
    
    return W

# Example usage
z = np.array([2.0, 1.0, 0.5])
probs = softmax(z)
print("Softmax probabilities:", probs)
print("Sum:", np.sum(probs))  # Should be 1.0
```

---

## 5. Deep Feedforward Neural Network (MLP)

```python
import numpy as np

def relu(z):
    """ReLU activation."""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU derivative."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Sigmoid derivative (given activation output)."""
    return a * (1 - a)

class SimpleMLP:
    """Simple 2-layer MLP for binary classification."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        """Forward propagation."""
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        
        # Layer 2 (output)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.output = sigmoid(self.z2)
        
        return self.output
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation."""
        n_samples = X.shape[0]
        
        # Output layer gradient
        dz2 = self.output - y.reshape(-1, 1)
        dW2 = (1/n_samples) * self.h1.T @ dz2
        db2 = (1/n_samples) * np.sum(dz2, axis=0)
        
        # Hidden layer gradient
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * relu_derivative(self.z1)
        dW1 = (1/n_samples) * X.T @ dz1
        db1 = (1/n_samples) * np.sum(dz1, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the network."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-15) + 
                               (1-y) * np.log(1-output + 1e-15))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions."""
        return (self.forward(X) >= 0.5).astype(int)

# Example: XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

mlp = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=1)
mlp.train(X_xor, y_xor, epochs=1000, learning_rate=0.5)
print("XOR predictions:", mlp.predict(X_xor).flatten())
```

---

## 6. CNN with Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_lenet(input_shape=(32, 32, 1), num_classes=10):
    """Create LeNet-like architecture."""
    model = Sequential([
        # Conv Layer 1
        Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Conv Layer 2
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """Create simple CNN for MNIST."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Compile and train
model = create_simple_cnn()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

---

## 7. Transfer Learning with Keras

```python
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def create_transfer_model(base_model_name='vgg16', num_classes=10, freeze_base=True):
    """
    Create transfer learning model.
    
    Parameters:
    -----------
    base_model_name : str - 'vgg16' or 'resnet50'
    num_classes : int - number of output classes
    freeze_base : bool - whether to freeze pre-trained layers
    """
    # Load pre-trained model
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(224, 224, 3))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
    
    # Freeze base layers if specified
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Usage
model = create_transfer_model('vgg16', num_classes=5, freeze_base=True)
model.compile(optimizer=Adam(lr=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

---

## 8. Utility Functions

```python
import numpy as np

# ============ LOSS FUNCTIONS ============

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def binary_cross_entropy(y_pred, y_true):
    """Binary Cross-Entropy loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_pred, y_true):
    """Categorical Cross-Entropy loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ============ METRICS ============

def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return np.mean(y_pred == y_true)

def precision(y_pred, y_true):
    """Calculate precision (binary)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-15)

def recall(y_pred, y_true):
    """Calculate recall (binary)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-15)

def f1_score(y_pred, y_true):
    """Calculate F1 score."""
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * (p * r) / (p + r + 1e-15)

def confusion_matrix(y_pred, y_true):
    """Create confusion matrix."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])

# ============ CNN UTILITIES ============

def compute_output_size(input_size, kernel_size, padding=0, stride=1):
    """Compute output size after convolution or pooling."""
    return int((input_size + 2 * padding - kernel_size) // stride + 1)

def compute_total_params(layer_sizes, include_bias=True):
    """
    Compute total trainable parameters.
    
    Parameters:
    -----------
    layer_sizes : list - [input_dim, hidden1, hidden2, ..., output_dim]
    include_bias : bool
    """
    total = 0
    for i in range(len(layer_sizes) - 1):
        weights = layer_sizes[i] * layer_sizes[i + 1]
        bias = layer_sizes[i + 1] if include_bias else 0
        total += weights + bias
    return total

# Example
layer_sizes = [784, 256, 128, 10]
print(f"Total params (with bias): {compute_total_params(layer_sizes)}")
print(f"Total params (no bias): {compute_total_params(layer_sizes, include_bias=False)}")
```

---

## 9. Quick Reference Snippets

### One-Hot Encoding
```python
def one_hot_encode(y, n_classes):
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot
```

### Data Preprocessing
```python
def normalize(X):
    """Min-max normalization to [0, 1]."""
    return (X - X.min()) / (X.max() - X.min() + 1e-15)

def standardize(X):
    """Z-score normalization."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)
```

### Train/Test Split
```python
def train_test_split(X, y, test_size=0.2, shuffle=True):
    """Simple train/test split."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

---

**🎯 These templates cover all exam coding patterns!**
