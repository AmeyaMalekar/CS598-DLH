import os
import sys

# Add CFVAE directory to Python path
sys.path.append(os.path.abspath('CFVAE'))

import torch
import pandas as pd
import numpy as np
from counterfactual_evaluation import CounterfactualEvaluator
from model import CFVAE
from torch.serialization import add_safe_globals

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the trained CFVAE model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Add CFVAE to safe globals for loading
    add_safe_globals(['model.CFVAE'])
    
    try:
        # Try loading the entire model
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded model")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Initializing new model with default parameters...")
        
        # Model parameters from our training
        feat_dim = 144  # Feature dimension from our data
        emb_dim1 = 144  # Embedding dimension from training
        mlp_inpemb = 72  # MLP input embedding dimension
        f_dim1 = 36  # First layer hidden units
        f_dim2 = 18  # Second layer hidden units
        
        # Initialize model with same architecture as training
        model = CFVAE(
            feat_dim=feat_dim,
            emb_dim1=emb_dim1,
            _mlp_dim1=9,
            _mlp_dim2=9,
            _mlp_dim3=9,
            mlp_inpemb=mlp_inpemb,
            f_dim1=f_dim1,
            f_dim2=f_dim2
        )
        
        # Try loading just the state dict
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
                print("Successfully loaded state dict")
        except Exception as e:
            print(f"Error loading state dict: {str(e)}")
    
    model.to(device)
    model.eval()
    
    return model

def load_data(data_dir: str) -> tuple:
    """
    Load the test data for evaluation.
    
    Args:
        data_dir: Directory containing the processed data
        
    Returns:
        Tuple of (test_data, test_labels, feature_names)
    """
    # Load test data
    X_test = pd.read_csv(os.path.join(data_dir, 'X2_curated_test.csv'), index_col=0)
    Y_test = pd.read_csv(os.path.join(data_dir, 'Y2_test.csv'), index_col=0)
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    # Convert to numpy arrays
    X_test = X_test.values.astype(np.float32)
    Y_test = Y_test.values.astype(np.float32)
    
    # Reshape data to match model input format
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Y_test is already in the correct shape (batch_size, 2) for binary classification
    
    # Convert to torch tensors
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)
    
    return X_test, Y_test, feature_names

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_path = "CFVAE/model/VAE_CF/multitask_rank/intervention_vaso/Adam/vae_epochs_100embed_144lr_0.001losswt_[1, 100].pt"
    data_dir = "processed_data"
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load data
    print("Loading data...")
    test_data, test_labels, feature_names = load_data(data_dir)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = CounterfactualEvaluator(model, device, feature_names)
    
    # Generate and evaluate counterfactuals
    print("Generating counterfactuals...")
    examples = evaluator.generate_counterfactual_examples(
        test_data.to(device),
        test_labels.to(device),
        num_examples=5
    )
    
    # Print evaluation metrics
    print("\nCounterfactual Evaluation Results:")
    print("=" * 50)
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print("-" * 30)
        print("Metrics:")
        for metric, value in example['metrics'].items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for feat, val in value.items():
                    print(f"  {feat}: {val:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
    
    # Plot counterfactual changes
    print("\nGenerating visualizations...")
    evaluator.plot_counterfactual_changes(examples, 'counterfactual_analysis')

if __name__ == "__main__":
    main() 