import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CFVAE
from utils import train_vae, evaluate_vae

"""
Script to train CFVAE with alternative hyperparameters:
- Larger batch size (64 vs original 32)
- Higher learning rate (5e-3 vs original 1e-3)
"""

# Modified hyperparameters
lr = 5e-3  # Increased from 1e-3
bs = 64    # Increased from 32
opt = 'Adam'
epochs = 100

# Original architecture parameters
emb_dim1 = 144
mlp_inpemb = 72
f_dim1 = 36
f_dim2 = 18
loss_wt = [1, 100]

dataset_splits = 'train', 'val', 'test'
intervention = 'vaso'

device = torch.device('cpu')
print(f"\nUsing device: {device}")

model_name = f'vae_bs{bs}_lr{lr}_epochs_{epochs}embed_{emb_dim1}'

# Loading the data
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

criterion_cf = nn.CrossEntropyLoss()
model = CFVAE(feat_dim, emb_dim1, 9, 9, 9, mlp_inpemb, f_dim1, f_dim2)
model = model.to(device)

opt_fn = {'adam': optim.Adam, 'sgd': optim.SGD}[opt.lower()]
optimizer = opt_fn(model.parameters(), lr)

criterion = nn.MSELoss()
criterion_x = nn.L1Loss()

# Create output directories
output_dir = f'output/VAE_CF_alt/intervention_{intervention}/{opt}/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'logs/VAE_CF_alt/intervention_{intervention}/{opt}/', exist_ok=True)
os.makedirs(f'model/VAE_CF_alt/intervention_{intervention}/{opt}/', exist_ok=True)

paths = {i: os.path.join(i, f'VAE_CF_alt/intervention_{intervention}/{opt}/')
         for i in ('logs', 'output', 'model')}

writer = SummaryWriter(paths['logs'] + model_name)

best_val_loss = float("inf")
best_val_cf_auc = 0

# Training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    tot_log_loss, train_loss_vae, train_loss_cf = train_vae(device, model, optimizer, criterion, criterion_cf,
                                                            criterion_x, loss_wt, bs, lr, epoch, epochs,
                                                            X2['train'], Y2['train'])
    writer.add_scalar('Loss/train_vae', train_loss_vae, epoch)
    writer.add_scalar('Loss/train_cf', train_loss_cf, epoch)
    
    val_loss, val_loss_vae, val_loss_cf, acc_cf, auc_cf, _ = evaluate_vae(device, model, optimizer, criterion,
                                                                          criterion_cf, criterion_x, loss_wt, bs, lr,
                                                                          epoch, epochs, X2['val'], Y2['val'])
    writer.add_scalar('Loss/val_vae', val_loss_vae, epoch)
    writer.add_scalar('Loss/val_cf', val_loss_cf, epoch)
    print('-' * 95)
    print('|end of epoch {:3d}| time: {:5.2f}s| valid loss {:5.2f} | valid acc {:5.2f} | valid auc {:5.2f} | '.format(
        epoch, (time.time() - epoch_start_time), val_loss, acc_cf, auc_cf))
    print('-' * 95)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(model.state_dict(), os.path.join(paths['model'], f'{model_name}.pt'))

# Final evaluation
test_loss, _, _, acc_cf, auc_cf, conf_cf = evaluate_vae(device, best_model, optimizer, criterion, criterion_cf,
                                                        criterion_x, loss_wt, bs, lr, epochs, epochs, X2['test'],
                                                        Y2['test'])
print('\nTest Results:')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {acc_cf:.4f}')
print(f'Test AUC: {auc_cf:.4f}') 