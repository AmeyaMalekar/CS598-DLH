import os
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

class MIMICDemoPreprocessor:
    def __init__(self, data_dir='mimic-iv-clinical-database-demo-2.2'):
        """Initialize the preprocessor with the path to the MIMIC-IV demo dataset."""
        self.data_dir = Path(data_dir)
        self.icu_dir = self.data_dir / 'icu'
        self.hosp_dir = self.data_dir / 'hosp'
        
        # Define vital signs and parameters to extract
        self.vital_signs = {
            'Heart Rate': ['Heart Rate', 'HR'],
            'Blood Pressure': ['Systolic BP', 'Diastolic BP', 'Mean BP'],
            'Temperature': ['Temperature'],
            'Respiratory Rate': ['Respiratory Rate', 'RR'],
            'SpO2': ['SpO2'],
            'GCS': ['GCS Total']
        }
        
        # Define intervention types
        self.interventions = {
            'vaso': ['Vasopressor', 'Norepinephrine', 'Epinephrine', 'Dopamine'],
            'vent': ['Mechanical Ventilation', 'Ventilator']
        }
        
        # Create output directory structure
        self.output_dir = Path('processed_data')
        self.output_dir.mkdir(exist_ok=True)
        
        # Define number of hours to consider
        self.num_hours = 24  # We'll use 24 hours of data for each stay
        
    def load_data(self):
        """Load the necessary tables from the MIMIC-IV demo dataset."""
        print("Loading data...")
        
        # Load ICU stays
        self.icustays = pd.read_csv(self.icu_dir / 'icustays.csv.gz', compression='gzip')
        
        # Load chart events (vital signs)
        self.chartevents = pd.read_csv(self.icu_dir / 'chartevents.csv.gz', compression='gzip')
        
        # Load input events (interventions)
        self.inputevents = pd.read_csv(self.icu_dir / 'inputevents.csv.gz', compression='gzip')
        
        # Load d_items for mapping itemids to names
        self.d_items = pd.read_csv(self.icu_dir / 'd_items.csv.gz', compression='gzip')
        
        print("Data loading complete.")
        
    def extract_vital_signs(self):
        """Extract and process vital signs from chartevents."""
        print("Extracting vital signs...")
        
        # Merge chartevents with d_items to get readable names
        chartevents_with_names = pd.merge(
            self.chartevents,
            self.d_items[['itemid', 'label']],
            on='itemid',
            how='left'
        )
        
        # Group by stay_id and time to get hourly measurements
        vital_signs = []
        for stay_id in self.icustays['stay_id'].unique():
            stay_data = chartevents_with_names[chartevents_with_names['stay_id'] == stay_id].copy()
            
            if stay_data.empty:
                continue
                
            # Convert charttime to datetime and set as index
            stay_data.loc[:, 'charttime'] = pd.to_datetime(stay_data['charttime'])
            stay_data.set_index('charttime', inplace=True)
            
            # Extract each vital sign
            stay_vitals = {}
            for sign, aliases in self.vital_signs.items():
                sign_data = stay_data[stay_data['label'].isin(aliases)]
                if not sign_data.empty:
                    # Take mean of measurements within each hour
                    hourly_mean = sign_data.resample('h')['valuenum'].mean()
                    
                    # Ensure we have exactly num_hours of data
                    if len(hourly_mean) > self.num_hours:
                        hourly_mean = hourly_mean[-self.num_hours:]  # Take most recent hours
                    elif len(hourly_mean) < self.num_hours:
                        # Pad with zeros if we have fewer hours
                        padding = pd.Series([0] * (self.num_hours - len(hourly_mean)), 
                                          index=pd.date_range(start=hourly_mean.index[0], 
                                                            periods=self.num_hours - len(hourly_mean), 
                                                            freq='h'))
                        hourly_mean = pd.concat([hourly_mean, padding])
                    
                    stay_vitals[sign] = hourly_mean.values
            
            if stay_vitals:
                vital_signs.append({
                    'stay_id': stay_id,
                    'vitals': stay_vitals
                })
        
        self.vital_signs_data = pd.DataFrame(vital_signs)
        print("Vital signs extraction complete.")
        
    def extract_interventions(self):
        """Extract intervention labels from inputevents."""
        print("Extracting interventions...")
        
        # Merge inputevents with d_items
        inputevents_with_names = pd.merge(
            self.inputevents,
            self.d_items[['itemid', 'label']],
            on='itemid',
            how='left'
        )
        
        # Create intervention labels
        intervention_labels = []
        for stay_id in self.icustays['stay_id'].unique():
            stay_data = inputevents_with_names[inputevents_with_names['stay_id'] == stay_id]
            
            # Check for each intervention type
            interventions = {}
            for intv_type, aliases in self.interventions.items():
                has_intervention = stay_data['label'].isin(aliases).any()
                interventions[intv_type] = 1 if has_intervention else 0
            
            if interventions:
                intervention_labels.append({
                    'stay_id': stay_id,
                    **interventions
                })
        
        self.intervention_labels = pd.DataFrame(intervention_labels)
        print("Intervention extraction complete.")
        
    def preprocess_features(self):
        """Preprocess the extracted features."""
        print("Preprocessing features...")
        
        # Combine vital signs into feature matrix
        features = []
        labels = []
        
        for _, row in self.vital_signs_data.iterrows():
            stay_id = row['stay_id']
            vitals = row['vitals']
            
            # Get corresponding intervention labels
            intv_labels = self.intervention_labels[self.intervention_labels['stay_id'] == stay_id]
            if intv_labels.empty:
                continue
                
            # Create feature vector
            feature_vector = []
            for sign in self.vital_signs.keys():
                if sign in vitals:
                    feature_vector.extend(vitals[sign])
                else:
                    # Fill missing values with zeros
                    feature_vector.extend([0] * self.num_hours)
                    
            features.append(feature_vector)
            labels.append(intv_labels[['vaso', 'vent']].values[0])
            
        # Convert to numpy arrays
        self.X = np.array(features, dtype=np.float32)
        self.Y = np.array(labels, dtype=np.int32)
        
        # Reshape for model input
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
        
        print("Feature preprocessing complete.")
        
    def split_data(self):
        """Split data into train/val/test sets."""
        print("Splitting data...")
        
        # First split: train + (val + test)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=42
        )
        
        # Second split: val and test
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42
        )
        
        self.splits = {
            'train': (X_train, Y_train),
            'val': (X_val, Y_val),
            'test': (X_test, Y_test)
        }
        
        print("Data splitting complete.")
        
    def save_data(self):
        """Save processed data in both pickle and CSV formats."""
        print("Saving data...")
        
        for split_name, (X, Y) in self.splits.items():
            # Save as pickle
            pickle_path = self.output_dir / f'{split_name}.pickle'
            with open(pickle_path, 'wb') as f:
                pickle.dump((X, None, None, None, Y), f)
                
            # Save as CSV
            X_df = pd.DataFrame(X.reshape(X.shape[0], -1))
            Y_df = pd.DataFrame(Y)
            
            X_path = self.output_dir / f'X2_curated_{split_name}.csv'
            Y_path = self.output_dir / f'Y2_{split_name}.csv'
            
            X_df.to_csv(X_path)
            Y_df.to_csv(Y_path)
            
        print("Data saving complete.")
        
    def process(self):
        """Run the complete preprocessing pipeline."""
        self.load_data()
        self.extract_vital_signs()
        self.extract_interventions()
        self.preprocess_features()
        self.split_data()
        self.save_data()
        
        print("\nPreprocessing complete!")
        print(f"Processed data saved in: {self.output_dir}")
        print("\nData shapes:")
        for split_name, (X, Y) in self.splits.items():
            print(f"{split_name}: X shape: {X.shape}, Y shape: {Y.shape}")

if __name__ == "__main__":
    preprocessor = MIMICDemoPreprocessor()
    preprocessor.process() 