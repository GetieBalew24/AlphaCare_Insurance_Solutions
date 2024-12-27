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
    def plot_bivariate_analysis(self, df, col1, col2, group_col):
        """
        Plots scatter plots and correlation matrices for bivariate analysis.

        This function performs a bivariate analysis by visualizing the relationship 
        between two numerical columns using a scatter plot, grouped by a categorical column. 
        It also computes and visualizes the correlation matrix for the two numerical columns.

        Parameters:
        df : pandas.DataFrame
            The input DataFrame containing the data.
        col1 : str
            The name of the first numerical column to analyze.
        col2 : str
            The name of the second numerical column to analyze.
        group_col : str
            The name of the categorical column used for grouping in the scatter plot.

        Outputs:
        Scatter plot : matplotlib plot
            A scatter plot visualizing the relationship between col1 and col2, grouped by group_col.
        Correlation matrix : heatmap
            A heatmap of the correlation matrix for col1 and col2.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col1, y=col2, hue=group_col, data=df)
        plt.title(f'Scatter Plot of {col1} vs {col2} by {group_col}')
        plt.show()

        corr_matrix = df[[col1, col2]].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='YlGn')
        plt.title('Correlation Matrix')
        plt.show()
     # Scatter plot functions to visualize relationships
    def scatter_plot(self, df, col_x, col_y):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col_x, y=col_y, data=df)
        plt.title(f'Scatter Plot of {col_x} vs. {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
