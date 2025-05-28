import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Import R² score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the data
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'  # Replace with your file path
data = pd.read_csv(file_path, sep=',', header=None)

# Correct label generation
labels = []
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0]  # Concentration levels

# Repeat each concentration value 300 times to match data points
for concentration in concentrations:
    labels.extend([concentration] * 300)

# If the number of data points isn't a multiple of 300, slice to match exactly the data length
labels = labels[:data.shape[0]]

# Convert labels to a numpy array
labels = np.array(labels)

# Normalize the sensor data (scaling to range [0, 1])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Check shape after scaling
print(f"Scaled data shape: {data_scaled.shape}")

# Since we have 3600 samples, the data shape will be (3600, 12)
X = data_scaled[:3600]  # Use the first 3600 data points

# Create labels (concentrations) corresponding to the 3600 data points
y = labels[:3600]

# Check the shape of X and y to ensure they are correct
print(f"X shape: {X.shape}")  # (3600, 12)
print(f"y shape: {y.shape}")  # (3600,)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Define the deep neural network model with multiple hidden layers and dropout
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)  # Dropout layer to prevent overfitting

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.fc4(x)
        return x


# Hyperparameters
input_size = 12  # Number of sensor features (columns)
hidden_sizes = [128, 256, 64, 32]  # Size of hidden layers
output_size = 1  # Regression task: predict a single concentration value
learning_rate = 0.001
num_epochs = 4000

# Initialize the model, loss function, and optimizer
model = DNNModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
criterion = nn.L1Loss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# List to store training losses and test losses for plotting
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute the loss
    loss = criterion(y_pred.squeeze(), y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Save the training loss for plotting
    train_losses.append(loss.item())

    # Evaluate the model on the test set after each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        test_loss = criterion(y_pred_test.squeeze(), y_test_tensor)

    # Save the test loss for plotting
    test_losses.append(test_loss.item())

    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Plot the training and test loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.grid(True)
plt.legend()
plt.show()

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test.squeeze(), y_test_tensor)
    print(f"Test Loss (final): {test_loss.item():.4f}")

# Convert predictions and true values to numpy for comparison
y_pred_test = y_pred_test.squeeze().numpy()
y_test = y_test_tensor.numpy()

# Compute the R² score
r2 = r2_score(y_test, y_pred_test)
print(f"R² Score: {r2:.4f}")

# Print some example predictions vs true values
for i in range(5):
    print(f"True: {y_test[i]}, Predicted: {y_pred_test[i]}")

# Plot true vs predicted concentrations
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Concentration', color='blue', marker='o')
plt.plot(y_pred_test, label='Predicted Concentration', color='red', marker='x')
plt.xlabel('Test Sample')
plt.ylabel('Concentration')
plt.title('True vs Predicted Concentration')
plt.legend()
plt.grid(True)
plt.show()

