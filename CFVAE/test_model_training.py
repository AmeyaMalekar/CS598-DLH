import os
import torch
import numpy as np
from model import CFVAE
from utils import train_vae, evaluate_vae
from torch.utils.data import TensorDataset, DataLoader

def test_model_setup():
    """Test the model initialization and forward pass"""
    print("\nTesting Model Setup...")
    print("-" * 50)
    
    # Get model parameters from main_vaeCF.py
    from main_vaeCF import (
        model, device, optimizer, criterion, criterion_cf, criterion_x,
        loss_wt, bs, lr, epochs, X2, Y2, feat_dim
    )
    
    # Test model architecture
    print("\nModel Architecture:")
    print(f"Feature dimension: {feat_dim}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    
    # Test forward pass
    print("\nTesting Forward Pass...")
    try:
        # Create a test batch with correct shape (batch_size, seq_len, 1)
        test_batch = torch.randn(bs, feat_dim, 1).float().to(device)
        output = model(test_batch)
        print("✓ Forward pass successful")
        print(f"Output shapes: {[o.shape for o in output]}")
        
        # Test with actual data batch
        X_train = torch.FloatTensor(X2['train']).to(device)
        Y_train = torch.FloatTensor(Y2['train']).to(device)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
        # Get one batch
        X_batch, _ = next(iter(train_loader))
        output = model(X_batch)
        print("\n✓ Forward pass with actual data successful")
        print(f"Output shapes with actual data: {[o.shape for o in output]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True

def test_training():
    """Test a small training session"""
    print("\nTesting Training...")
    print("-" * 50)
    
    # Get model parameters from main_vaeCF.py
    from main_vaeCF import (
        model, device, optimizer, criterion, criterion_cf, criterion_x,
        loss_wt, bs, lr, epochs, X2, Y2
    )
    
    try:
        # Run one epoch of training
        print("\nRunning one training epoch...")
        tot_log_loss, train_loss_vae, train_loss_cf = train_vae(
            device, model, optimizer, criterion, criterion_cf,
            criterion_x, loss_wt, bs, lr, 1, epochs,
            X2['train'], Y2['train']
        )
        
        print("✓ Training successful")
        print(f"Total loss: {tot_log_loss:.4f}")
        print(f"VAE loss: {train_loss_vae:.4f}")
        print(f"CF loss: {train_loss_cf:.4f}")
        
        # Run validation
        print("\nRunning validation...")
        val_loss, val_loss_vae, val_loss_cf, acc_cf, auc_cf, _ = evaluate_vae(
            device, model, optimizer, criterion, criterion_cf,
            criterion_x, loss_wt, bs, lr, 1, epochs,
            X2['val'], Y2['val']
        )
        
        print("✓ Validation successful")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {acc_cf:.4f}")
        print(f"Validation AUC: {auc_cf:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Model Testing...")
    print("=" * 50)
    
    # Test model setup
    if test_model_setup():
        print("\nModel setup successful, proceeding to training test...")
        test_training()
    else:
        print("\nModel setup failed, please check the configuration.") 