"""
Medical Surge Prediction System using PyTorch
==============================================

This system predicts which medical conditions will surge on given dates
using historical admission data with temporal and seasonal features.

Author: AI Assistant
Created: January 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MedicalDataset(Dataset):
    """Custom Dataset for medical surge prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MedicalSurgePredictor(nn.Module):
    """
    Neural Network for predicting medical condition surges
    
    Architecture:
    - Input: Temporal features (day, month, season, etc.)
    - Hidden layers with dropout for regularization
    - Output: Probability of surge for each medical condition
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(MedicalSurgePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Multi-label classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MedicalSurgeAnalyzer:
    """Main class for medical surge prediction analysis"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.condition_columns = []
        self.feature_columns = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the medical data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Convert dates (handle mixed formats with error handling)
        def parse_date_safely(date_str):
            """Parse dates with multiple format attempts"""
            if pd.isna(date_str):
                return None
            try:
                # Try common formats
                formats = ['%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y']
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                # If all formats fail, try pandas default parsing
                return pd.to_datetime(date_str, dayfirst=False)
            except:
                return None
        
        print("Parsing dates...")
        self.df['D.O.A'] = self.df['D.O.A'].apply(parse_date_safely)
        self.df['D.O.D'] = self.df['D.O.D'].apply(parse_date_safely)
        
        # Remove rows with unparseable dates
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['D.O.A', 'D.O.D'])
        print(f"Removed {initial_len - len(self.df)} rows with invalid dates")
        
        # Extract temporal features from admission date
        self.df['year'] = self.df['D.O.A'].dt.year
        self.df['month'] = self.df['D.O.A'].dt.month
        self.df['day'] = self.df['D.O.A'].dt.day
        self.df['day_of_week'] = self.df['D.O.A'].dt.dayofweek
        self.df['day_of_year'] = self.df['D.O.A'].dt.dayofyear
        self.df['week_of_year'] = self.df['D.O.A'].dt.isocalendar().week
        self.df['quarter'] = self.df['D.O.A'].dt.quarter
        
        # Create seasonal features
        self.df['season'] = self.df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring  
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })
        
        # Create cyclical features for time
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # Identify medical condition columns (binary indicators)
        self.condition_columns = [
            'SMOKING ', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
            'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',
            'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',
            'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT',
            'PSVT', 'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
            'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
            'PULMONARY EMBOLISM', 'CHEST INFECTION'
        ]
        
        # Feature columns for prediction
        self.feature_columns = [
            'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter', 'season',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'AGE'
        ]
        
        # Handle missing values and ensure numeric types
        for col in self.condition_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        print("Data preprocessing completed!")
        
    def create_surge_data(self, window_days=7, surge_threshold=1.5):
        """
        Create surge prediction data by aggregating conditions over time windows
        
        Args:
            window_days: Days to aggregate data over
            surge_threshold: Multiplier above average to consider as surge
        """
        print(f"Creating surge data with {window_days}-day windows...")
        
        # Create date range for the entire dataset
        min_date = self.df['D.O.A'].min()
        max_date = self.df['D.O.A'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        surge_data = []
        
        for date in date_range:
            # Get data for the current window
            window_start = date
            window_end = date + timedelta(days=window_days-1)
            
            window_data = self.df[
                (self.df['D.O.A'] >= window_start) & 
                (self.df['D.O.A'] <= window_end)
            ]
            
            if len(window_data) == 0:
                continue
                
            # Create features for this date
            features = {
                'date': date,
                'month': date.month,
                'day_of_week': date.weekday(),
                'day_of_year': date.timetuple().tm_yday,
                'week_of_year': date.isocalendar()[1],
                'quarter': (date.month - 1) // 3 + 1,
                'season': {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                          6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[date.month],
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'day_sin': np.sin(2 * np.pi * date.weekday() / 7),
                'day_cos': np.cos(2 * np.pi * date.weekday() / 7),
                'avg_age': window_data['AGE'].mean()
            }
            
            # Calculate condition counts for this window
            for condition in self.condition_columns:
                if condition in window_data.columns:
                    count = window_data[condition].sum()
                    features[f'{condition}_count'] = count
                else:
                    features[f'{condition}_count'] = 0
            
            surge_data.append(features)
        
        surge_df = pd.DataFrame(surge_data)
        
        # Calculate historical averages for surge detection
        for condition in self.condition_columns:
            count_col = f'{condition}_count'
            if count_col in surge_df.columns:
                # Calculate rolling average (excluding current window)
                surge_df[f'{condition}_avg'] = surge_df[count_col].rolling(
                    window=30, min_periods=10
                ).mean().shift(1)
                
                # Define surge as count > threshold * average
                surge_df[f'{condition}_surge'] = (
                    surge_df[count_col] > 
                    surge_threshold * surge_df[f'{condition}_avg']
                ).astype(int)
        
        # Remove rows with NaN averages (early dates)
        surge_df = surge_df.dropna()
        
        print(f"Created {len(surge_df)} surge prediction samples")
        return surge_df
    
    def prepare_training_data(self):
        """Prepare features and targets for training"""
        print("Preparing training data...")
        
        # Create surge data
        self.surge_df = self.create_surge_data()
        
        # Prepare features
        feature_cols = [
            'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter', 'season',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'avg_age'
        ]
        
        X = self.surge_df[feature_cols].values
        
        # Prepare targets (surge indicators)
        target_cols = [f'{condition}_surge' for condition in self.condition_columns 
                      if f'{condition}_surge' in self.surge_df.columns]
        y = self.surge_df[target_cols].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Number of conditions: {y.shape[1]}")
        
        return X_train, X_test, y_train, y_test, target_cols
    
    def build_model(self, input_size, output_size):
        """Build the neural network model"""
        print("Building model...")
        
        hidden_sizes = [128, 64, 32]  # Deep network for complex patterns
        self.model = MedicalSurgePredictor(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=0.3
        )
        
        print(f"Model architecture: {input_size} -> {' -> '.join(map(str, hidden_sizes))} -> {output_size}")
        return self.model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the model"""
        print("Training model...")
        
        # Create data loaders
        train_dataset = MedicalDataset(X_train, y_train)
        test_dataset = MedicalDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.BCELoss()  # Binary cross entropy for multi-label
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    test_loss += loss.item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        print("Training completed!")
        return train_losses, test_losses
    
    def predict_surges(self, target_date, days_ahead=7):
        """
        Predict surges for a specific date or date range
        
        Args:
            target_date: Target date for prediction (string or datetime)
            days_ahead: Number of days to predict ahead
        
        Returns:
            Dictionary with predictions for each condition
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        predictions = {}
        
        for day_offset in range(days_ahead):
            pred_date = target_date + timedelta(days=day_offset)
            
            # Create features for prediction date
            features = np.array([[
                pred_date.month,
                pred_date.weekday(),
                pred_date.timetuple().tm_yday,
                pred_date.isocalendar()[1],
                (pred_date.month - 1) // 3 + 1,
                {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[pred_date.month],
                np.sin(2 * np.pi * pred_date.month / 12),
                np.cos(2 * np.pi * pred_date.month / 12),
                np.sin(2 * np.pi * pred_date.weekday() / 7),
                np.cos(2 * np.pi * pred_date.weekday() / 7),
                65.0  # Average age assumption
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                pred_probs = self.model(features_tensor).numpy()[0]
            
            # Store predictions
            date_str = pred_date.strftime('%Y-%m-%d')
            predictions[date_str] = {}
            
            for i, condition in enumerate(self.condition_columns):
                if i < len(pred_probs):
                    predictions[date_str][condition] = float(pred_probs[i])
        
        return predictions
    
    def save_model(self, model_path='medical_surge_model.pth'):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'condition_columns': self.condition_columns,
            'feature_columns': self.feature_columns
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='medical_surge_model.pth'):
        """Load a pre-trained model"""
        checkpoint = torch.load(model_path)
        
        # Rebuild model architecture
        input_size = 11  # Number of features
        output_size = len(checkpoint['condition_columns'])
        self.model = self.build_model(input_size, output_size)
        
        # Load saved components
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.condition_columns = checkpoint['condition_columns']
        self.feature_columns = checkpoint['feature_columns']
        
        print(f"Model loaded from {model_path}")
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in the data"""
        print("Analyzing seasonal patterns...")
        
        if self.surge_df is None:
            print("No surge data available. Run prepare_training_data() first.")
            return
        
        # Monthly patterns
        monthly_stats = self.surge_df.groupby('month').agg({
            f'{condition}_count': 'mean' for condition in self.condition_columns[:10]
        }).round(2)
        
        print("\nAverage monthly condition counts (top 10 conditions):")
        print(monthly_stats)
        
        # Seasonal patterns
        seasonal_stats = self.surge_df.groupby('season').agg({
            f'{condition}_count': 'mean' for condition in self.condition_columns[:10]
        }).round(2)
        
        seasonal_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
        seasonal_stats.index = seasonal_stats.index.map(seasonal_names)
        
        print("\nAverage seasonal condition counts (top 10 conditions):")
        print(seasonal_stats)
        
        return monthly_stats, seasonal_stats
    
def main():
    """Main execution function"""
    print("Medical Surge Prediction System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = MedicalSurgeAnalyzer('overcrowding.csv')
    
    # Load and preprocess data
    analyzer.load_and_preprocess_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, target_cols = analyzer.prepare_training_data()
    
    # Build model
    model = analyzer.build_model(input_size=X_train.shape[1], output_size=y_train.shape[1])
    
    # Train model
    train_losses, test_losses = analyzer.train_model(X_train, y_train, X_test, y_test, epochs=50)
    
    # Save model
    analyzer.save_model('medical_surge_model.pth')
    
    # Analyze patterns
    analyzer.analyze_seasonal_patterns()
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Predict for next week
    predictions = analyzer.predict_surges('2024-01-15', days_ahead=7)
    
    for date, preds in predictions.items():
        print(f"\nDate: {date}")
        # Show top 5 highest risk conditions
        sorted_conditions = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        print("Top 5 surge risk conditions:")
        for condition, prob in sorted_conditions[:5]:
            print(f"  {condition}: {prob:.3f}")
    
    print("\nModel training and prediction completed successfully!")
    print("Files created:")
    print("- medical_surge_model.pth: Trained PyTorch model")
    print("- medical_surge_predictor.py: Complete source code")

if __name__ == "__main__":
    main()