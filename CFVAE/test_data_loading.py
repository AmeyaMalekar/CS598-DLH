import os
import numpy as np
import pandas as pd

def test_data_loading():
    """Test loading and processing of our data files"""
    data_path = 'processed_data/'
    dataset_splits = ['train', 'val', 'test']
    
    print("Testing data loading...")
    print("-" * 50)
    
    for split in dataset_splits:
        print(f"\nTesting {split} set:")
        
        # Test feature data loading
        try:
            X_path = os.path.join(data_path, f'X2_curated_{split}.csv')
            X = pd.read_csv(X_path, index_col=0)
            print(f"✓ Successfully loaded X2_curated_{split}.csv")
            print(f"  Shape: {X.shape}")
            print(f"  Data types: {X.dtypes.unique()}")
            print(f"  Missing values: {X.isnull().sum().sum()}")
            
            # Handle missing values
            X = X.fillna(0)  # Replace missing values with 0
            print(f"  Missing values after handling: {X.isnull().sum().sum()}")
        except Exception as e:
            print(f"✗ Error loading X2_curated_{split}.csv: {e}")
        
        # Test label data loading
        try:
            Y_path = os.path.join(data_path, f'Y2_{split}.csv')
            Y = pd.read_csv(Y_path, index_col=0)
            print(f"✓ Successfully loaded Y2_{split}.csv")
            print(f"  Shape: {Y.shape}")
            print(f"  Unique labels: {Y.iloc[:, 0].unique()}")
            print(f"  Label distribution: {Y.iloc[:, 0].value_counts()}")
            
            # Take only the first column for binary classification
            Y = Y.iloc[:, 0]
            print(f"  Y shape after taking first column: {Y.shape}")
        except Exception as e:
            print(f"✗ Error loading Y2_{split}.csv: {e}")
        
        # Test reshaping
        try:
            X_np = X.values.astype(np.float32)
            Y_np = Y.values.astype(np.float32)
            
            X_reshaped = np.reshape(X_np, (X_np.shape[0], X_np.shape[1], 1))
            Y_reshaped = np.reshape(Y_np, (Y_np.shape[0], 1))
            
            print(f"✓ Successfully reshaped data")
            print(f"  X shape after reshape: {X_reshaped.shape}")
            print(f"  Y shape after reshape: {Y_reshaped.shape}")
            
            # Verify the shapes match what the model expects
            if X_reshaped.shape[1] == 144:  # Our feature dimension
                print("  ✓ Feature dimension matches expected size")
            else:
                print(f"  ✗ Feature dimension mismatch: expected 144, got {X_reshaped.shape[1]}")
                
            if Y_reshaped.shape[1] == 1:
                print("  ✓ Label dimension matches expected size")
            else:
                print(f"  ✗ Label dimension mismatch: expected 1, got {Y_reshaped.shape[1]}")
        except Exception as e:
            print(f"✗ Error reshaping data: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_data_loading() 