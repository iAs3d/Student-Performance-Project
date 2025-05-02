# Student Performance Predictor

This project uses machine learning to predict student exam scores (math, reading, and writing) based on demographic and behavioral factors. The model takes inputs such as gender, race/ethnicity, parental education level, lunch type, and test preparation course completion to predict performance outcomes.

## Project Structure

```
├── app.py                 # Streamlit web application
├── train.py               # Script to train models and generate visualizations
├── requirements.txt       # Project dependencies
├── StudentsPerformance.csv # Dataset
├── src/
│   └── model.py           # Model implementation
├── models/                # Saved trained models (generated after training)
└── plots/                 # Visualizations (generated after training)
```

## Features

The model predicts student performance based on the following features:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type (standard or free/reduced)
- Test Preparation Course Completion

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

To train the models and generate visualizations:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train regression models for math, reading, and writing scores
- Save the trained models to the `models/` directory
- Generate visualizations in the `plots/` directory

### Running the Web Application

To launch the Streamlit web application:

```bash
streamlit run app.py
```

The application provides three main pages:
1. **Prediction**: Enter student information to predict their exam scores
2. **Data Exploration**: Explore the dataset with visualizations
3. **Model Information**: View model architecture and feature importance

## Model Details

The project uses Gradient Boosting Regression models to predict student scores. The data preprocessing includes:
- One-hot encoding for categorical features
- Train-test split (80% training, 20% testing)

Model performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Dataset

The dataset contains information about student performance in exams, including demographic information and test scores. It includes the following columns:
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- math score
- reading score
- writing score