import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the model class
from src.model import StudentPerformanceModel

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Title and description
st.title("Student Performance Predictor")
st.markdown("""
This application predicts a student's performance in math, reading, and writing exams 
based on various demographic and behavioral factors.
""")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Prediction", "Data Exploration", "Model Information"])

# Initialize model
model = StudentPerformanceModel(data_path="StudentsPerformance.csv")

# Load data
try:
    data = model.load_data()
    st.sidebar.success("Data loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    data = None

# Check if models exist, if not train them
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

if not os.path.exists(os.path.join(models_dir, "math_model.joblib")) or \
   not os.path.exists(os.path.join(models_dir, "reading_model.joblib")) or \
   not os.path.exists(os.path.join(models_dir, "writing_model.joblib")):
    
    with st.sidebar.status("Training models..."):
        # Preprocess data
        model.preprocess_data()
        # Train models
        results = model.train_models()
        # Save models
        model.save_models(output_dir=models_dir)
    st.sidebar.success("Models trained and saved!")
else:
    # Load existing models
    model.load_models(input_dir=models_dir)
    st.sidebar.success("Models loaded successfully!")

# Prediction page
if page == "Prediction":
    st.header("Predict Student Performance")
    st.markdown("Enter student information to predict their exam scores.")
    
    # Create form for input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=["female", "male"])
            race_ethnicity = st.selectbox("Race/Ethnicity", 
                                         options=["group A", "group B", "group C", "group D", "group E"])
            parental_education = st.selectbox("Parental Level of Education", 
                                             options=["some high school", "high school", "some college", 
                                                      "associate's degree", "bachelor's degree", "master's degree"])
        
        with col2:
            lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"])
            test_prep = st.selectbox("Test Preparation Course", options=["none", "completed"])
        
        submit_button = st.form_submit_button("Predict Scores")
    
    # Make prediction when form is submitted
    if submit_button:
        # Create input data dictionary
        input_data = {
            "gender": gender,
            "race/ethnicity": race_ethnicity,
            "parental level of education": parental_education,
            "lunch": lunch,
            "test preparation course": test_prep
        }
        
        # Get prediction
        try:
            predictions = model.predict(input_data)
            
            # Display predictions
            st.subheader("Predicted Scores")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Math Score", f"{predictions['math']:.1f}")
            
            with col2:
                st.metric("Reading Score", f"{predictions['reading']:.1f}")
            
            with col3:
                st.metric("Writing Score", f"{predictions['writing']:.1f}")
            
            # Calculate average score
            avg_score = (predictions['math'] + predictions['reading'] + predictions['writing']) / 3
            
            # Display performance category
            st.subheader("Overall Performance")
            if avg_score >= 80:
                st.success("Excellent Performance (80-100)")
            elif avg_score >= 70:
                st.info("Good Performance (70-79)")
            elif avg_score >= 60:
                st.warning("Average Performance (60-69)")
            else:
                st.error("Needs Improvement (Below 60)")
            
            # Display a bar chart of the scores
            fig, ax = plt.subplots(figsize=(10, 6))
            subjects = ['Math', 'Reading', 'Writing']
            scores = [predictions['math'], predictions['reading'], predictions['writing']]
            
            bars = ax.bar(subjects, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylim(0, 100)
            ax.set_ylabel('Score')
            ax.set_title('Predicted Scores by Subject')
            
            # Add score labels on top of bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{score:.1f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if data is not None:
        # Display basic information about the dataset
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {data.shape[0]}")
        st.write(f"Number of features: {data.shape[1]}")
        
        # Display first few rows of the dataset
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(data.describe())
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Distribution of scores
        st.markdown("### Distribution of Scores")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.histplot(data['math score'], kde=True, ax=axes[0])
        axes[0].set_title('Math Scores')
        
        sns.histplot(data['reading score'], kde=True, ax=axes[1])
        axes[1].set_title('Reading Scores')
        
        sns.histplot(data['writing score'], kde=True, ax=axes[2])
        axes[2].set_title('Writing Scores')
        
        st.pyplot(fig)
        
        # Correlation heatmap
        st.markdown("### Correlation Between Scores")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        score_cols = ['math score', 'reading score', 'writing score']
        corr = data[score_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        
        st.pyplot(fig)
        
        # Categorical feature analysis
        st.markdown("### Impact of Categorical Features on Scores")
        
        categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 
                               'lunch', 'test preparation course']
        
        selected_feature = st.selectbox("Select Feature", categorical_features)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.boxplot(x=selected_feature, y='math score', data=data, ax=axes[0])
        axes[0].set_title(f'Math Scores by {selected_feature}')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x=selected_feature, y='reading score', data=data, ax=axes[1])
        axes[1].set_title(f'Reading Scores by {selected_feature}')
        axes[1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x=selected_feature, y='writing score', data=data, ax=axes[2])
        axes[2].set_title(f'Writing Scores by {selected_feature}')
        axes[2].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
    else:
        st.error("Data not available for exploration.")

# Model Information page
elif page == "Model Information":
    st.header("Model Information")
    
    st.markdown("""
    ### Model Architecture
    
    This application uses **Gradient Boosting Regression** models to predict student scores in math, reading, and writing.
    
    The models take the following features as input:
    - Gender
    - Race/Ethnicity
    - Parental Level of Education
    - Lunch Type (standard or free/reduced)
    - Test Preparation Course Completion
    
    ### Data Preprocessing
    
    - Categorical features are encoded using One-Hot Encoding
    - Data is split into training (80%) and testing (20%) sets
    
    ### Model Performance
    
    The models are evaluated using the following metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R² Score
    """)
    
    # If models are trained, show feature importance
    if all(model.models.values()):
        st.subheader("Feature Importance")
        st.markdown("The charts below show the importance of each feature in predicting student scores.")
        
        try:
            feature_importance = model.evaluate_feature_importance()
            
            for subject, importance_df in feature_importance.items():
                st.markdown(f"#### {subject.capitalize()} Score Prediction")
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.barplot(x='importance', y='feature', data=importance_df.head(10), ax=ax)
                ax.set_title(f'Top 10 Features for {subject.capitalize()} Score Prediction')
                
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error evaluating feature importance: {e}")
    else:
        st.warning("Models need to be trained to display feature importance.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About

This application was built to predict student performance based on various factors.

""")