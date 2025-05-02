import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class StudentPerformanceModel:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.models = {
            'math': None,
            'reading': None,
            'writing': None
        }
        self.preprocessor = None
        
    def load_data(self, data_path=None):
        """Load the dataset from CSV file"""
        if data_path:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("Data path must be provided")
            
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        return self.data
    
    def explore_data(self):
        """Perform basic exploratory data analysis"""
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        # Display basic information
        print("\nBasic Information:")
        print(self.data.info())
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes,
            'missing_values': self.data.isnull().sum().to_dict()
        }
    
    def preprocess_data(self):
        """Preprocess the data for model training"""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Define features and targets
        X = self.data.drop(['math score', 'reading score', 'writing score'], axis=1)
        y_math = self.data['math score']
        y_reading = self.data['reading score']
        y_writing = self.data['writing score']
        
        # Combine targets into a DataFrame
        y = pd.DataFrame({
            'math': y_math,
            'reading': y_reading,
            'writing': y_writing
        })
        
        # Define categorical and numerical features
        categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 
                               'lunch', 'test preparation course']
        
        # Create preprocessor
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.X = X
        self.y = y
        
        return X, y
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train regression models for each subject"""
        if self.X is None or self.y is None:
            raise ValueError("Data must be preprocessed first")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Train models for each subject
        subjects = ['math', 'reading', 'writing']
        results = {}
        
        for subject in subjects:
            print(f"\nTraining model for {subject} score prediction...")
            
            # Create pipeline with preprocessor and model
            model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(random_state=random_state))
            ])
            
            # Train model
            model.fit(X_train, y_train[subject])
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            mse = mean_squared_error(y_test[subject], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[subject], y_pred)
            r2 = r2_score(y_test[subject], y_pred)
            
            print(f"  Mean Squared Error: {mse:.2f}")
            print(f"  Root Mean Squared Error: {rmse:.2f}")
            print(f"  Mean Absolute Error: {mae:.2f}")
            print(f"  R² Score: {r2:.2f}")
            
            # Store model
            self.models[subject] = model
            
            # Store results
            results[subject] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model
            }
        
        return results
    
    def evaluate_feature_importance(self):
        """Evaluate feature importance for each model"""
        if not all(self.models.values()):
            raise ValueError("Models must be trained first")
        
        feature_importance = {}
        
        for subject, model in self.models.items():
            # Get feature names after one-hot encoding
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 
                                   'lunch', 'test preparation course']
            
            # Get the one-hot encoder from the pipeline
            ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            
            # Get feature names
            feature_names = ohe.get_feature_names_out(categorical_features)
            
            # Get feature importances
            importances = model.named_steps['regressor'].feature_importances_
            
            # Create a DataFrame of feature importances
            feature_importance[subject] = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk"""
        if not all(self.models.values()):
            raise ValueError("Models must be trained first")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for subject, model in self.models.items():
            model_path = os.path.join(output_dir, f"{subject}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Model for {subject} saved to {model_path}")
        
        return True
    
    def load_models(self, input_dir='models'):
        """Load trained models from disk"""
        # Load each model
        for subject in self.models.keys():
            model_path = os.path.join(input_dir, f"{subject}_model.joblib")
            if os.path.exists(model_path):
                self.models[subject] = joblib.load(model_path)
                print(f"Model for {subject} loaded from {model_path}")
            else:
                print(f"Model file {model_path} not found")
        
        return all(self.models.values())
    
    def predict(self, input_data):
        """Make predictions using trained models"""
        if not all(self.models.values()):
            raise ValueError("Models must be trained first")
        
        # Convert input data to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Make predictions for each subject
        predictions = {}
        for subject, model in self.models.items():
            predictions[subject] = model.predict(input_data)[0]
        
        return predictions