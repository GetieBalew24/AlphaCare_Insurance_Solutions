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