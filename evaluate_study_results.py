import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from CFVAE.utils import evaluate_vae
import os

def load_model_results(model_path):
    """Load model results from the output file"""
    results = {}
    with open(model_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Test accuracy cf:' in line:
                results['test_acc'] = float(line.split(':')[1])
            elif 'Test AUC cf:' in line:
                results['test_auc'] = float(line.split(':')[1])
            elif 'Best val accuracy CF:' in line:
                results['val_acc'] = float(line.split(':')[1])
            elif 'Best val AUC CF:' in line:
                results['val_auc'] = float(line.split(':')[1])
    return results

def compare_with_paper(our_results, paper_results):
    """Compare our results with paper's reported results"""
    print("\nComparison with Paper Results:")
    print("=" * 50)
    print(f"{'Metric':<20} {'Our Results':<15} {'Paper Results':<15} {'Difference':<15}")
    print("-" * 50)
    
    for metric in ['test_acc', 'test_auc', 'val_acc', 'val_auc']:
        our_value = our_results.get(metric, 0)
        paper_value = paper_results.get(metric, 0)
        diff = our_value - paper_value
        print(f"{metric:<20} {our_value:<15.4f} {paper_value:<15.4f} {diff:<15.4f}")

def analyze_counterfactuals(model, test_data, device):
    """Analyze the quality of generated counterfactuals"""
    model.eval()
    with torch.no_grad():
        # Generate counterfactuals for a sample of test data
        sample_data = test_data[:10]  # Analyze first 10 samples
        original_predictions = model(sample_data)
        counterfactuals = model.generate_counterfactual(sample_data)
        
        # Calculate counterfactual effectiveness
        effectiveness = []
        for orig, cf in zip(original_predictions, counterfactuals):
            effectiveness.append(torch.abs(orig - cf).mean().item())
        
        return np.mean(effectiveness)

def main():
    # Load our model results
    model_path = "CFVAE/output/VAE_CF/multitask_rank/intervention_vaso/Adam/output_vae_epochs_100embed_144lr_0.001losswt_[1, 100].txt"
    our_results = load_model_results(model_path)
    
    # Paper's reported results (to be filled with actual values from paper)
    paper_results = {
        'test_acc': 0.85,  # Placeholder - update with actual paper values
        'test_auc': 0.75,  # Placeholder - update with actual paper values
        'val_acc': 0.82,   # Placeholder - update with actual paper values
        'val_auc': 0.78    # Placeholder - update with actual paper values
    }
    
    # Compare results
    compare_with_paper(our_results, paper_results)
    
    # Additional analysis
    print("\nAdditional Analysis:")
    print("=" * 50)
    print(f"Test Accuracy: {our_results['test_acc']:.4f}")
    print(f"Test AUC: {our_results['test_auc']:.4f}")
    print(f"Validation Accuracy: {our_results['val_acc']:.4f}")
    print(f"Validation AUC: {our_results['val_auc']:.4f}")

if __name__ == "__main__":
    main() 