# Import necessary libraries
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class StatisticalModel:
    def __init__(self, data):
        self.data = data

    def fit_model_per_zipcode(self, data):
        """
        Fits a linear regression model for each zipcode to predict total claims.
        Parameters:
        - data: pd.DataFrame with features and target ('TotalClaims').
        Returns:
        - models: Dictionary with zipcodes as keys and trained models as values.
        - predictions: Dictionary with zipcodes as keys and predictions as values.
        """
        # Store models and predictions
        models = {}
        predictions = {}
        # Group by Zipcode
        grouped = data.groupby('PostalCode')
        
        for zipcode, group in grouped:
            # Define features and target
            X = group.drop(columns=['TotalClaims', 'PostalCode'])
            y = group['TotalClaims']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and fit the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store the model and predictions
            models[zipcode] = model
            predictions[zipcode] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'mse': mean_squared_error(y_test, y_pred)
            }
            
            # Print the performance for each zipcode
            print(f"Zipcode: {zipcode}")
            print(f"Mean Squared Error: {predictions[zipcode]['mse']:.4f}")
        
        return models, predictions