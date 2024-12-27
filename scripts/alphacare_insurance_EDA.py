import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceEDA:
    def __init__(self, data):
        self.data = data
    
    # Data Summarization
    def summarize_data(self, df):
        """
        Provides descriptive statistics and data types.
        """
        print("Data Summary:\n", df.describe())
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())
     # Univariate Analysis: Plotting histograms and bar charts
    def plot_histograms(self,df, numerical_cols):
        """
        Plots histograms for numerical columns.
        """
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=90, ha='right')
            plt.show()
    def plot_bar_charts(self,df,categorical_cols):
        """
        Plots bar charts for categorical columns.
        """
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=df)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=90, ha='right')
            plt.show()