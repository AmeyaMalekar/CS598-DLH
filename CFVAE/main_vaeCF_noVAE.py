import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import get_batch, final_loss

# Hyperparameters
lr = 1e-3
bs = 32
opt = 'Adam'
epochs = 100

# Architecture parameters
input_dim = 144
hidden_dim = 72
output_dim = 1

# Loss weights (only using counterfactual loss)
loss_wt = [0, 100, 0.1]  # [VAE loss, counterfactual loss, sparsity loss]

# Load datasets
train_data = pd.read_csv('processed_data/train_data.csv')
val_data = pd.read_csv('processed_data/val_data.csv')
test_data = pd.read_csv('processed_data/test_data.csv')

# Print data statistics
print("\nData Statistics:")
print(f"Training set: {len(train_data)} samples, {train_data.shape[1]-1} features")
print(f"Label distribution: {train_data['label'].value_counts().to_dict()}")
print(f"\nValidation set: {len(val_data)} samples, {val_data.shape[1]-1} features")
print(f"Label distribution: {val_data['label'].value_counts().to_dict()}")
print(f"\nTest set: {len(test_data)} samples, {test_data.shape[1]-1} features")
print(f"Label distribution: {test_data['label'].value_counts().to_dict()}")

# Prepare data
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_val = val_data.drop('label', axis=1).values
y_val = val_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs)
test_loader = DataLoader(test_dataset, batch_size=bs)

# Define the model without VAE
class DirectCounterfactual(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DirectCounterfactual, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.mlp(x)
    
    def generate_counterfactual(self, x, target_class):
        x.requires_grad_(True)
        output = self.forward(x)
        loss = nn.BCELoss()(output, target_class)
        loss.backward()
        
        # Generate counterfactual by moving in the direction of the gradient
        with torch.no_grad():
            counterfactual = x - 0.1 * x.grad
            counterfactual = torch.clamp(counterfactual, 0, 1)
        return counterfactual

# Initialize model
model = DirectCounterfactual(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# Create output directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Training loop
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target.unsqueeze(1)).item()
            val_preds.extend(output.numpy())
            val_targets.extend(target.numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_targets, np.array(val_preds) > 0.5)
    val_auc = roc_auc_score(val_targets, val_preds)
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Valid Loss: {val_loss:.2f}, Valid Accuracy: {val_acc:.2f}, Valid AUC: {val_auc:.2f}')
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model_noVAE.pt')

# Test evaluation
model.load_state_dict(torch.load('models/best_model_noVAE.pt'))
model.eval()
test_loss = 0
test_preds = []
test_targets = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target.unsqueeze(1)).item()
        test_preds.extend(output.numpy())
        test_targets.extend(target.numpy())

test_loss /= len(test_loader)
test_acc = accuracy_score(test_targets, np.array(test_preds) > 0.5)
test_auc = roc_auc_score(test_targets, test_preds)

print("\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save results
results = {
    'test_loss': test_loss,
    'test_accuracy': test_acc,
    'test_auc': test_auc
}
pd.DataFrame([results]).to_csv('results/noVAE_results.csv', index=False) 