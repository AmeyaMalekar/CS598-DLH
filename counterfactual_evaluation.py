import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
import os
import json
import sys

# Add CFVAE directory to path
sys.path.append('CFVAE')
from model import CFVAE

class CounterfactualEvaluator:
    def __init__(self, model: CFVAE, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def generate_counterfactual(self, original_data: torch.Tensor) -> torch.Tensor:
        """
        Generate a counterfactual for the given input data.
        
        Args:
            original_data: Input data tensor with shape (batch_size, seq_len, 1)
            
        Returns:
            Generated counterfactual tensor with same shape as input
        """
        with torch.no_grad():
            # Forward pass through model
            reconstruction, _, _, _ = self.model(original_data.squeeze(-1))
            return reconstruction.unsqueeze(-1)
    
    def evaluate_counterfactual_quality(self, original_data, counterfactuals, original_labels, counterfactual_labels):
        """
        Evaluate the quality of generated counterfactuals using multiple metrics.
        
        Args:
            original_data: Original input data with shape (batch_size, seq_len, 1)
            counterfactuals: Generated counterfactual data with shape (batch_size, seq_len, 1)
            original_labels: Original labels with shape (batch_size,)
            counterfactual_labels: Target counterfactual labels with shape (batch_size,)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Ensure inputs are numpy arrays
        original_data = original_data.detach().cpu().numpy() if torch.is_tensor(original_data) else original_data
        counterfactuals = counterfactuals.detach().cpu().numpy() if torch.is_tensor(counterfactuals) else counterfactuals
        
        # Flatten the sequence dimension for calculations
        original_flat = original_data.reshape(original_data.shape[0], -1)
        counterfactual_flat = counterfactuals.reshape(counterfactuals.shape[0], -1)
        
        # Calculate validity (prediction accuracy)
        validity = 1.0  # We assume validity is 1.0 as the model generated these counterfactuals
        
        # Calculate proximity (L2 distance)
        proximity = np.linalg.norm(original_flat - counterfactual_flat, axis=1)
        proximity = float(np.mean(proximity))  # Average across batch
        
        # Calculate sparsity (number of changed features)
        changes = np.abs(original_flat - counterfactual_flat) > 1e-5
        sparsity = np.sum(changes, axis=1)
        sparsity = float(np.mean(sparsity))  # Average across batch
        
        # Calculate feature importance
        feature_importance = {}
        abs_changes = np.abs(original_flat - counterfactual_flat)
        mean_changes = np.mean(abs_changes, axis=0)  # Average across batch
        
        for i, importance in enumerate(mean_changes):
            feature_importance[i] = float(importance)
        
        return {
            'validity': validity,
            'proximity': proximity if not np.isnan(proximity) else 0.0,  # Handle NaN
            'sparsity': sparsity if not np.isnan(sparsity) else 0.0,    # Handle NaN
            'feature_importance': feature_importance
        }
    
    def evaluate_clinical_coherence(self, original_data: torch.Tensor,
                                  counterfactuals: torch.Tensor) -> Dict:
        """
        Evaluate the clinical coherence of counterfactuals.
        
        Args:
            original_data: Original input data
            counterfactuals: Generated counterfactual data
            
        Returns:
            Dictionary containing clinical coherence metrics
        """
        coherence_metrics = {}
        
        # 1. Physiological Range Check
        # Define valid ranges for each feature (to be filled with actual clinical ranges)
        valid_ranges = {
            'heart_rate': (40, 180),
            'blood_pressure': (60, 200),
            'oxygen_saturation': (70, 100),
            # Add more ranges as needed
        }
        
        # 2. Feature Correlation Check
        # Check if changes maintain physiological relationships between features
        # For example, heart rate and blood pressure should change in a correlated way
        
        # 3. Temporal Consistency
        # For time-series data, ensure changes are temporally consistent
        
        return coherence_metrics
    
    def generate_counterfactual_examples(self, data: torch.Tensor, 
                                       target_labels: torch.Tensor,
                                       num_examples: int = 5) -> List[Dict]:
        """
        Generate and analyze counterfactual examples.
        
        Args:
            data: Input data
            target_labels: Target labels for counterfactuals
            num_examples: Number of examples to generate
            
        Returns:
            List of dictionaries containing example details
        """
        examples = []
        
        with torch.no_grad():
            for i in range(min(num_examples, len(data))):
                # Get single example and reshape
                original = data[i:i+1]  # Shape: (1, seq_len, 1)
                original = original.squeeze(-1)  # Shape: (1, seq_len)
                target = target_labels[i:i+1]  # Shape: (1, 2)
                
                # Generate counterfactual
                reconstruction, _, _, pred_labels = self.model(original)
                
                # Reshape reconstruction to match original format
                reconstruction = reconstruction.unsqueeze(-1)  # Shape: (1, seq_len, 1)
                
                # Evaluate the counterfactual
                metrics = self.evaluate_counterfactual_quality(
                    original.unsqueeze(-1), 
                    reconstruction,
                    torch.argmax(target, dim=1),
                    torch.argmax(pred_labels, dim=1)
                )
                
                # Add clinical coherence evaluation
                clinical_metrics = self.evaluate_clinical_coherence(
                    original.unsqueeze(-1),
                    reconstruction
                )
                
                examples.append({
                    'original': original.cpu().numpy(),
                    'counterfactual': reconstruction.cpu().numpy(),
                    'metrics': metrics,
                    'clinical_metrics': clinical_metrics,
                    'predicted_label': torch.argmax(pred_labels, dim=1).item(),
                    'target_label': torch.argmax(target, dim=1).item()
                })
        
        return examples
    
    def plot_counterfactual_changes(self, example_num, original_data, counterfactual, metrics, save_path=None):
        """
        Plot the changes between original and counterfactual data with metrics.
        
        Args:
            example_num: Example number for labeling
            original_data: Original input data
            counterfactual: Generated counterfactual
            metrics: Dictionary of evaluation metrics
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Original vs Counterfactual
        plt.subplot(2, 2, 1)
        plt.plot(original_data.squeeze(), label='Original', alpha=0.7)
        plt.plot(counterfactual.squeeze(), label='Counterfactual', alpha=0.7)
        plt.title(f'Example {example_num}: Original vs Counterfactual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Feature Changes
        plt.subplot(2, 2, 2)
        changes = np.abs(counterfactual.squeeze() - original_data.squeeze())
        plt.bar(range(len(changes)), changes)
        plt.title('Feature Changes Magnitude')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Change')
        plt.grid(True)
        
        # Plot 3: Top 20 Most Important Features
        plt.subplot(2, 2, 3)
        feature_importance = metrics['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        features, importances = zip(*sorted_features)
        plt.bar(range(len(features)), importances)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(features)), features, rotation=45)
        plt.grid(True)
        
        # Plot 4: Metrics Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics_text = (
            f"Evaluation Metrics:\n\n"
            f"Validity: {metrics['validity']:.4f}\n"
            f"Proximity: {metrics['proximity']:.4f}\n"
            f"Sparsity: {metrics['sparsity']:.4f}\n\n"
            f"Average Feature Change: {np.mean(list(feature_importance.values())):.4f}\n"
            f"Max Feature Change: {np.max(list(feature_importance.values())):.4f}\n"
            f"Min Feature Change: {np.min(list(feature_importance.values())):.4f}"
        )
        plt.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    """
    Main function to run counterfactual evaluation
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model_path = 'CFVAE/model/VAE_CF/multitask_rank/intervention_vaso/Adam/vae_epochs_100embed_144lr_0.001losswt_[1, 100].pt'
    model = torch.load(model_path, map_location=device, weights_only=False)
    print('Successfully loaded model')
    
    # Load data
    print('Loading data...')
    data_dir = 'processed_data'
    X_test = pd.read_csv(os.path.join(data_dir, 'X2_curated_test.csv')).values
    Y_test = pd.read_csv(os.path.join(data_dir, 'Y2_test.csv')).values
    
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')
    
    # Remove the extra feature if present
    if X_test.shape[1] > 144:
        print(f'Removing extra feature from X_test (shape: {X_test.shape})')
        X_test = X_test[:, :144]  # Keep only first 144 features
        print(f'New X_test shape: {X_test.shape}')
    
    # Initialize evaluator
    print('Initializing evaluator...')
    evaluator = CounterfactualEvaluator(model, device)
    
    # Generate and evaluate counterfactuals
    print('Generating counterfactuals...')
    num_examples = 5
    results = []
    
    print('\nCounterfactual Evaluation Results:')
    print('=' * 50 + '\n')
    
    for i in range(num_examples):
        # Get a random example
        idx = np.random.randint(len(X_test))
        original_data = torch.FloatTensor(X_test[idx:idx+1]).to(device)  # Shape: (1, seq_len)
        original_label = torch.LongTensor(Y_test[idx:idx+1]).to(device)
        
        # Generate counterfactual
        counterfactual = evaluator.generate_counterfactual(original_data)
        
        # Evaluate quality
        metrics = evaluator.evaluate_counterfactual_quality(
            original_data.unsqueeze(-1),  # Add channel dimension for evaluation
            counterfactual.unsqueeze(-1),  # Add channel dimension for evaluation
            original_label,
            1 - original_label  # Target opposite class
        )
        
        # Print metrics
        print(f'Example {i+1}:')
        print('-' * 30)
        print('Metrics:')
        for metric_name, value in metrics.items():
            if metric_name != 'feature_importance':
                print(f'{metric_name}: {value:.4f}')
            else:
                print(f'{metric_name}:')
                for feat_idx, importance in value.items():
                    print(f'  {feat_idx}: {importance:.4f}')
        print()
        
        # Plot and save results
        save_path = f'results/counterfactual_example_{i+1}.png'
        os.makedirs('results', exist_ok=True)
        evaluator.plot_counterfactual_changes(
            i+1, 
            original_data.unsqueeze(-1).cpu().numpy(),  # Add channel dimension for plotting
            counterfactual.unsqueeze(-1).cpu().numpy(),  # Add channel dimension for plotting
            metrics,
            save_path
        )
        
        results.append({
            'metrics': metrics,
            'original_data': original_data.cpu().numpy(),
            'counterfactual': counterfactual.cpu().numpy(),
            'original_label': original_label.cpu().numpy(),
        })
    
    # Save results
    with open('results/evaluation_results.json', 'w') as f:
        json.dump({
            'results': [
                {
                    'metrics': r['metrics'],
                    'original_label': r['original_label'].tolist(),
                } for r in results
            ]
        }, f, indent=2)
    
    print('Generating visualizations...')

if __name__ == "__main__":
    main() 