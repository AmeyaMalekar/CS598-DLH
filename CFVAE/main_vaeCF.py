import os
import pickle
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
Script to load an MLP model trained to predict if intervention is required in the next 24 hours and train a VAE to generate counterfactual examples

Variables:
    intervention: 'vaso','vent' (intervention that we are predicting)
    loss_wt: [VAE loss, CF loss] weighting the overall loss function of the system 
    pt_modelname: pretrained intervention prediction model name
Hyperparameters:
    emb_dim1: Size of linear layers in the VAE  

    lr: learning rate
    bs: batch size
    epochs 

    Note: mlp_inpemb,f_dim1, f_dim2 values should be equal to a pre-trained intervention prediction model hyperparameters
"""

emb_dim1 = 144  # size of the outer hidden layer of the VAE (matching our feature dimension)
lr = 1e-3  # learning rate parameter for training
bs = 32  # batch size parameter for training
opt = 'Adam'  # optimizer used
epochs = 100  # number of epochs
loss_wt = [1, 100]  # weight of VAE loss (0) and weight of counterfactual loss (1)
mlp_inpemb = 72  # dimension of the word embedding (half of feature dimension)
f_dim1 = 36  # hidden units in first layer of MLP (half of mlp_inpemb)
f_dim2 = 18  # hidden units in second layer of MLP (half of f_dim1)

dataset_splits = 'train', 'val', 'test'
intervention = 'vaso'

# Set device to CPU since CUDA is not available
device = torch.device('cpu')
print(f"\nUsing device: {device}")

model_name = f'vae_epochs_{epochs}embed_{emb_dim1}lr_{lr}losswt_{loss_wt}'
pt_modelname = f'multitaskmlp_{mlp_inpemb}embed_{f_dim1}fc1_{f_dim2}fc2_epochs_32bs.lr.pt'

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

num_1 = len(np.where(Y2['train'][:, 0] == 1)[0])
num_0 = len(np.where(Y2['train'][:, 0] == 0)[0])
num = max(num_0, num_1)

criterion_cf = nn.CrossEntropyLoss()

model = CFVAE(feat_dim, emb_dim1, 9, 9, 9, mlp_inpemb, f_dim1, f_dim2)

opt_fn = {'adam': optim.Adam, 'sgd': optim.SGD}[opt.lower()]
optimizer = opt_fn(model.parameters(), lr)

criterion = nn.MSELoss()
criterion_x = nn.L1Loss()

# Create output directories if they don't exist
output_dir = 'output/VAE_CF/multitask_rank/intervention_{intervention}/{opt}/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('logs/VAE_CF/multitask_rank/intervention_{intervention}/{opt}/', exist_ok=True)
os.makedirs('model/VAE_CF/multitask_rank/intervention_{intervention}/{opt}/', exist_ok=True)

paths = {i: os.path.join(i, f'VAE_CF/multitask_rank/intervention_{intervention}/{opt}/')
         for i in ('logs', 'output', 'model')}

writer = SummaryWriter(paths['logs'] + model_name)

best_val_loss = float("inf")
best_val_cf_auc = 0

# Initialize model with random weights
model.to(device)

epoch_start_time = 0
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

test_loss, _, _, acc_cf, auc_cf, conf_cf = evaluate_vae(device, best_model, optimizer, criterion, criterion_cf,
                                                        criterion_x, loss_wt, bs, lr, epochs, epochs, X2['test'],
                                                        Y2['test'])
val_loss, _, _, val_acc_cf, val_auc_cf, val_conf_cf = evaluate_vae(device, best_model, optimizer, criterion,
                                                                   criterion_cf, criterion_x, loss_wt, bs, lr, epochs,
                                                                   epochs, X2['val'], Y2['val'])
print('=' * 95)
print(
    '|end of training {:3d}| time: {:5.2f}s| test loss {:5.2f} | test acc {:5.2f} | test auc {:5.2f} | '.format(epochs,
                                                                                                                time.time() - epoch_start_time,
                                                                                                                test_loss,
                                                                                                                acc_cf,
                                                                                                                auc_cf))

outfile = paths['output'] + 'output_' + model_name + '.txt'

if not os.path.exists(paths['output']):
    os.makedirs(paths['output'])

if not os.path.exists(paths['model']):
    os.makedirs(paths['model'])

torch.save(best_model, paths['model'] + model_name + '.pt')

with open(outfile, 'w') as f:
    f.write(model_name + '\n')
    f.write('Test accuracy cf:' + str(acc_cf) + '\n')
    f.write('Test AUC cf:' + str(auc_cf) + '\n')
    f.write('Conf cf:' + str(conf_cf) + '\n')

    f.write('Best val accuracy CF: ' + str(val_acc_cf) + '\n')
    f.write('Best val AUC CF:' + str(val_auc_cf) + '\n')
    f.write('Best val conf CF:' + str(val_conf_cf) + '\n')
