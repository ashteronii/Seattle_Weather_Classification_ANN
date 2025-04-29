"""
This script trains an Artificial Neural Network (ANN) on the Seattle Weather dataset
to classify the weather based on features like temperature, precipitation, and wind speed.

Steps:
1. Load and clean the dataset.
2. Preprocess the features and target variable.
3. Define a custom PyTorch dataset class for batching.
4. Build an ANN model for multi-class classification.
5. Train and evaluate the model using cross-entropy loss and accuracy metrics.
6. Visualize performance with confusion matrix and classification report.

The dataset is from Ananth R. on Kaggle, contains weather data for Seattle,
and has been cleaned for use.
Dataset citation: Ananth R. (2022). *seattle-weather*. Kaggle. https://www.kaggle.com/datasets/ananthr1/weather-prediction
"""

# Import necessary libraries.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report

# Load the cleaned Seattle Weather dataset.
weather = pd.read_csv('Seattle_Weather_Cleaned.csv')

# Extract features (drop 'weather', 'date', and 'Unnamed: 0').
X = weather.drop(columns=['weather', 'date', 'Unnamed: 0'])

# Extract the target variable ('weather').
y = weather['weather']

# Encode the target labels as integers.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data int training and validation sets (70%/30% split).
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the feature data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Define a custom Dataset class for PyTorch.
class SeattleWeatherDataset(Dataset):
    """
    A custom PyTorch Dataset for loading the Seattle Weather data.

    This class takes in feature data (X) and labels (y) and prepares them to
    be used by PyTorch DataLoaders for efficient mini-batch loading.

    Each data point is converted into a PyTorch tensor of the appropriate type:
    - Features (X) are stored as float32 tensors.
    - Labels (y) are stored as long integers (required for classification tasks).
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset instances.
train_dataset = SeattleWeatherDataset(X_train, y_train)
valid_dataset = SeattleWeatherDataset(X_valid, y_valid)

# Define the Artificial Neural Network model that will train the WeatherNet ANN on the Seattle Weather dataset.
class WeatherNet(nn.Module):
    """
    A feedforward Artificial Neural Network (ANN) for classifying Seattle weather.

    Architecture:
    - Input layer: expects 4 standardized features.
    - Hidden layer 1: 16 neurons, BatchNorm, and ReLU.
    - Hidden layer 2: 8 neurons, BatchNorm, and ReLU.
    - Output layer: size equal to the number of weather classes.

    Batch normalization is used after each hidden layer to stabilize
    and speed up training.
    """
    def __init__(self):
        super(WeatherNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)                 # First hidden layer.
        self.norm1 = nn.BatchNorm1d(16)                                  # Batch normalization layer.
        self.fc2 = nn.Linear(16, 8)                 # Second hidden layer.
        self.norm2 = nn.BatchNorm1d(8)                                   # Batch normalization layer
        self.fc3 = nn.Linear(8, len(label_encoder.classes_))   # Output layer.

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.fc3(x)
        return x

# Define a function to train the Artificial Neural Network.
def trainNN(epochs=50, batch_size=16, lr=0.001):
    """
    Trains the WeatherNet ANN using cross-entropy loss and the Adam optimizer.

    Parameters:
        epochs (int): Number of training iterations over the dataset.
        batch_size (int): Mini-batch size used for training.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the trained PyTorch model and the validation DataLoader.

    The function prints training and validation loss after each epoch for monitoring performance.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = WeatherNet()
    criterion = nn.CrossEntropyLoss()                        # Loss function for multi-class classification.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer.

    for epoch in range(epochs):
        # Training the Artificial Neural Network.
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

        # Validating the Artificial Neural Network.
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        print("-" * 40)

    return model, valid_loader

# Train the model and get the validation DataLoader.
model, valid_loader = trainNN(epochs=50)

# Evaluate the model on the validation set.
model.eval()  # Set the model to evaluation mode (important for layers like dropout, batch norm, etc.).
y_true = []   # To store actual labels.
y_pred = []   # To store predicted labels.

with torch.no_grad():                     # Disable gradient computation for evaluation to save memory.
    for X_batch, y_batch in valid_loader:
        outputs = model(X_batch)          # Get the model predictions for the batch.
        _, preds = torch.max(outputs, 1)  # Get the predicted class (index of max probability).

        # Convert tensors to numpy arrays for evaluation and store in lists.
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.numpy())

# Calculate and print the validation accuracy.
acc = accuracy_score(y_true, y_pred)     # Calculate the accuracy based on true and predicted labels.
print(f"Validation Accuracy: {acc:.4f}")

# Print the classification metrics (precision, recall, f1-score).
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Plot the confusion matrix.
cm = confusion_matrix(y_true, y_pred)                                                      # Calculate the confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)  # Display the matrix.
disp.plot()
plt.title('Confusion Matrix')
plt.show()