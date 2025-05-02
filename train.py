import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import StudentPerformanceModel

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize model
    model = StudentPerformanceModel(data_path="StudentsPerformance.csv")
    
    # Load and explore data
    data = model.load_data()
    exploration_results = model.explore_data()
    
    # Preprocess data
    X, y = model.preprocess_data()
    
    # Train models
    results = model.train_models(test_size=0.2, random_state=42)
    
    # Save models
    model.save_models(output_dir='models')
    
    # Generate and save visualizations
    generate_visualizations(data)
    
    # Evaluate feature importance
    feature_importance = model.evaluate_feature_importance()
    plot_feature_importance(feature_importance)
    
    print("\nTraining and analysis completed successfully!")
    print("Run 'streamlit run app.py' to start the web application.")

def generate_visualizations(data):
    """Generate and save visualizations for data analysis"""
    # Distribution of scores
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data['math score'], kde=True)
    plt.title('Math Scores Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(data['reading score'], kde=True)
    plt.title('Reading Scores Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(data['writing score'], kde=True)
    plt.title('Writing Scores Distribution')
    
    plt.tight_layout()
    plt.savefig('plots/score_distributions.png')
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    score_cols = ['math score', 'reading score', 'writing score']
    corr = data[score_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Scores')
    plt.tight_layout()
    plt.savefig('plots/score_correlations.png')
    
    # Box plots for categorical features
    categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 
                           'lunch', 'test preparation course']
    
    for feature in categorical_features:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(x=feature, y='math score', data=data)
        plt.title(f'Math Scores by {feature}')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        sns.boxplot(x=feature, y='reading score', data=data)
        plt.title(f'Reading Scores by {feature}')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        sns.boxplot(x=feature, y='writing score', data=data)
        plt.title(f'Writing Scores by {feature}')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        # Replace slashes with underscores in feature names for filenames
        safe_feature_name = feature.replace('/', '_')
        plt.savefig(f'plots/{safe_feature_name}_analysis.png')

def plot_feature_importance(feature_importance):
    """Plot and save feature importance for each subject"""
    for subject, importance_df in feature_importance.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(10))
        plt.title(f'Top 10 Features for {subject.capitalize()} Score Prediction')
        plt.tight_layout()
        plt.savefig(f'plots/{subject}_feature_importance.png')

if __name__ == "__main__":
    main()