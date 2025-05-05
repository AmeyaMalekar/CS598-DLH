import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import DirectCFModel

"""
Script to train a direct counterfactual generation model without VAE component.

Variables:
    intervention: 'vaso','vent' (intervention that we are predicting)
Hyperparameters:
    lr: learning rate
    bs: batch size
    epochs 
    mlp_inpemb: dimension of the word embedding
    f_dim1: hidden units in first layer of MLP
    f_dim2: hidden units in second layer of MLP
"""

lr = 1e-3  # learning rate parameter for training
bs = 32  # batch size parameter for training
opt = 'Adam'  # optimizer used
epochs = 100  # number of epochs
mlp_inpemb = 72  # dimension of the word embedding (half of feature dimension)
f_dim1 = 36  # hidden units in first layer of MLP (half of mlp_inpemb)
f_dim2 = 18  # hidden units in second layer of MLP (half of f_dim1)

dataset_splits = 'train', 'val', 'test'
intervention = 'vaso'

# Set device to CPU since CUDA is not available
device = torch.device('cpu')
print(f"\nUsing device: {device}")

model_name = f'directcf_epochs_{epochs}embed_{mlp_inpemb}'

# Loading the data from our preprocessed MIMIC-IV demo dataset
data_path = 'processed_data/'
X2 = {}
Y2 = {}

for s in dataset_splits:
    # Load feature data
    X2[s] = pd.read_csv(os.path.join(data_path, f'X2_curated_{s}.csv'), index_col=0)
    # Load label data
    Y2[s] = pd.read_csv(os.path.join(data_path, f'Y2_{s}.csv'), index_col=0)
    
    # Handle missing values in features
    X2[s] = X2[s].fillna(0)
    
    # Take only the first column for binary classification
    Y2[s] = Y2[s].iloc[:, 0]
    
    # Convert to numpy arrays and ensure proper types
    X2[s] = X2[s].values.astype(np.float32)
    Y2[s] = Y2[s].values.astype(np.float32)
    
    # Reshape data to match expected format
    X2[s] = np.reshape(X2[s], (X2[s].shape[0], X2[s].shape[1], 1))
    Y2[s] = np.reshape(Y2[s], (Y2[s].shape[0], 1))

feat_dim = X2['train'].shape[1]

# Print data statistics
print("\nData Statistics:")
print("-" * 50)
for s in dataset_splits:
    print(f"\n{s} set:")
    print(f"Number of samples: {X2[s].shape[0]}")
    print(f"Number of features: {X2[s].shape[1]}")
    print(f"Label distribution: {np.unique(Y2[s], return_counts=True)}")
print("-" * 50)

# Initialize model and optimizer
model = DirectCFModel(feat_dim, mlp_inpemb, f_dim1, f_dim2)
model = model.to(device)

opt_fn = {'adam': optim.Adam, 'sgd': optim.SGD}[opt.lower()]
optimizer = opt_fn(model.parameters(), lr)

criterion = nn.CrossEntropyLoss()

# Create output directories if they don't exist
output_dir = f'output/Direct_CF/intervention_{intervention}/{opt}/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'logs/Direct_CF/intervention_{intervention}/{opt}/', exist_ok=True)
os.makedirs(f'model/Direct_CF/intervention_{intervention}/{opt}/', exist_ok=True)

paths = {i: os.path.join(i, f'Direct_CF/intervention_{intervention}/{opt}/')
         for i in ('logs', 'output', 'model')}

writer = SummaryWriter(paths['logs'] + model_name)

best_val_loss = float("inf")
best_val_acc = 0

def train_epoch(model, optimizer, criterion, X, Y):
    model.train()
    total_loss = 0
    num_correct = 0
    num_total = 0
    
    for i in range(0, X.shape[0], bs):
        end_idx = min(i + bs, X.shape[0])
        data = torch.from_numpy(X[i:end_idx, :, 0])
        target = torch.from_numpy(Y[i:end_idx, 0])
        
        data = data.to(device)
        target = target.to(device)
        data = data.float()
        target = target.long()
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(output, 1)
        num_correct += (predicted == target).sum().item()
        num_total += target.shape[0]
    
    return total_loss / max(1, (X.shape[0] // bs)), num_correct / max(1, num_total)

def evaluate(model, criterion, X, Y):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            end_idx = min(i + bs, X.shape[0])
            data = torch.from_numpy(X[i:end_idx, :, 0])
            target = torch.from_numpy(Y[i:end_idx, 0])
            
            data = data.to(device)
            target = target.to(device)
            data = data.float()
            target = target.long()
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            num_correct += (predicted == target).sum().item()
            num_total += target.shape[0]
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / max(1, (X.shape[0] // bs)), num_correct / max(1, num_total), all_preds, all_targets

# Training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    
    # Training
    train_loss, train_acc = train_epoch(model, optimizer, criterion, X2['train'], Y2['train'])
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    
    # Validation
    val_loss, val_acc, _, _ = evaluate(model, criterion, X2['val'], Y2['val'])
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    print('-' * 95)
    print('|end of epoch {:3d}| time: {:5.2f}s| valid loss {:5.2f} | valid acc {:5.2f} |'.format(
        epoch, (time.time() - epoch_start_time), val_loss, val_acc))
    print('-' * 95)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(model.state_dict(), os.path.join(paths['model'], f'{model_name}.pt'))

# Final evaluation on test set
test_loss, test_acc, test_preds, test_targets = evaluate(best_model, criterion, X2['test'], Y2['test'])
print('\nTest Results:')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# Save final model
torch.save(best_model.state_dict(), os.path.join(paths['model'], f'{model_name}_final.pt')) 