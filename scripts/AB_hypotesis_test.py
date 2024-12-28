import pandas as pd
import scipy.stats as stats

class ABHypothesisTester:
    def __init__(self, data):
        self.data = data
    # Define control and test groups
    def create_groups(self, df, feature, group_A_values, group_B_values):
        group_A = df[df[feature].isin(group_A_values)]
        group_B = df[df[feature].isin(group_B_values)]
        return group_A, group_B
