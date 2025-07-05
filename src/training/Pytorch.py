import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load the feature-engineered dataset
df = pd.read_csv("data/processed/telco_churn_feature_engineered.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Calculate pos_weight for imbalanced dataset
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32)

# Define the neural network model
class ChurnModel(nn.Module):
    def __init__(self, input_size, num_hidden_layers=2, hidden_size=64):
        super(ChurnModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Hyperparameter tuning
best_precision = 0
best_params = {}
best_model_state = None

learning_rates = [0.01, 0.001, 0.0001]
num_hidden_layers_options = [2, 3, 4]
hidden_size_options = [32, 64, 128]

for lr in learning_rates:
    for num_layers in num_hidden_layers_options:
        for hidden_size in hidden_size_options:
            print(f"\nTraining with LR: {lr}, Layers: {num_layers}, Hidden Size: {hidden_size}")
            input_size = X_train.shape[1]
            model = ChurnModel(input_size, num_layers, hidden_size)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            epochs = 200 # Increased epochs
            batch_size = 64
            losses = []

            for epoch in range(epochs):
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    losses.append(loss.item())
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Evaluate on validation set
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_preds = torch.sigmoid(val_outputs).round()
                    val_precision = precision_score(y_val, val_preds)

                # Early stopping (simple version: save best model)
                if val_precision > best_precision:
                    best_precision = val_precision
                    best_params = {'lr': lr, 'num_layers': num_layers, 'hidden_size': hidden_size}
                    best_model_state = model.state_dict()

# Load the best model state
if best_model_state:
    model = ChurnModel(input_size, best_params['num_layers'], best_params['hidden_size'])
    model.load_state_dict(best_model_state)
    print(f"\nBest Model Parameters: {best_params}")
    print(f"Best Validation Precision: {best_precision:.4f}")
else:
    print("No best model found. Using default parameters.")

# Evaluate the best model on the test set
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_cls = torch.sigmoid(y_pred_logits).round()
    accuracy = accuracy_score(y_test, y_pred_cls)
    precision = precision_score(y_test, y_pred_cls)
    recall = recall_score(y_test, y_pred_cls)
    f1 = f1_score(y_test, y_pred_cls)

    print(f"\nFinal Test Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Save the best model
torch.save(model.state_dict(), "models/pytorch_model.pth")

# Plot and save the loss graph (for the last trained model in the loop, or you can modify to save for best model)
# Note: This will plot the loss of the *last* trained model in the loop, not necessarily the best one.
# To plot for the best model, you would need to store losses for each run and select the best one.
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss (Last Run)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()
plt.savefig("models/loss_graph.png")

