import pandas as pd
import scipy.stats as stats

class ABHypothesisTester:
    """    
    A class for conducting A/B hypothesis testing on different groups in a dataset.
    """  
    def __init__(self, data):
        self.data = data
    # Define control and test groups
    def create_groups(self, df, feature, group_A_values, group_B_values):
        """
        Creates two groups (A and B) based on specific values of a given feature.

        Parameters:
        ----------
        df : pandas.DataFrame
            The dataset to split into groups.
        feature : str
            The column name used to split the groups.
        group_A_values : list
            Values in the feature column to include in Group A.
        group_B_values : list
            Values in the feature column to include in Group B.
        """
        group_A = df[df[feature].isin(group_A_values)]
        group_B = df[df[feature].isin(group_B_values)]
        return group_A, group_B
    
    # Hypothesis Testing for Gender
    def create_gender_groups(self, df):
        """
        Creates two groups (A and B) based on the 'Gender' column.

        Parameters:
        ----------
        df : pandas.DataFrame
            The dataset to split into groups.

        Returns:
        -------
        tuple:
            Two DataFrames (Group A for 'Male', Group B for 'Female').
        """
        group_A = df[df['Gender'] == 'Male']
        group_B = df[df['Gender'] == 'Female']
        return group_A, group_B
     # Perform Hypothesis Testing for ZipCode
    def create_zipcode_groups(self, df):
        """
        Creates two groups (A and B) based on the 'PostalCode' column.
        The zip codes are split into two halves.

        Parameters:
        ----------
        df : pandas.DataFrame
            The dataset to split into groups.

        Returns:
        -------
        tuple:
            Two DataFrames (Group A, Group B).
        """
        # Define a threshold or specific zip codes for grouping
        zipcodes = df['PostalCode'].unique()
        mid_index = len(zipcodes) // 2
        group_A = df[df['PostalCode'].isin(zipcodes[:mid_index])]
        group_B = df[df['PostalCode'].isin(zipcodes[mid_index:])]
        return group_A, group_B
       # Hypothesis Testing Function
    def hypothesis_test(self, group_A, group_B, column, test_type='t'):
        """
        Performs hypothesis testing (t-test or chi-squared test) between two groups on a specified column.

        Parameters:
        ----------
        group_A : pandas.DataFrame
            The first group for hypothesis testing.
        group_B : pandas.DataFrame
            The second group for hypothesis testing.
        column : str
            The column name on which the hypothesis test is performed.
        test_type : str, optional
            The type of test to perform ('t' for t-test or 'chi2' for chi-squared test). Default is 't'.

        Returns:
        -------
        float:
            The p-value from the hypothesis test.

        Raises:
        -------
        ValueError:
            If an unsupported test type is provided.
        """
        if test_type == 't':
            # T-test for numerical data
            t_stat, p_value = stats.ttest_ind(group_A[column].dropna(), group_B[column].dropna())
        elif test_type == 'chi2':
            # Chi-Squared test for categorical data
            contingency_table = pd.crosstab(group_A[column].dropna(), group_B[column].dropna())
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        else:
            raise ValueError("Unsupported test type. Use 't' for t-test or 'chi2' for chi-squared test.")
        
        return p_value
    def report_results(self, p_value, test_name):
        """
        Reports the results of a hypothesis test based on the p-value.

        Parameters:
        ----------
        p_value : float
            The p-value obtained from the hypothesis test.
        test_name : str
            The name of the test being performed (e.g., "T-test for Gender").

        Returns:
        -------
        None:
            Prints the test result and conclusion (whether to reject or fail to reject the null hypothesis).

        Notes:
        ------
        A p-value threshold of 0.05 is used:
        - If p-value < 0.05, the null hypothesis is rejected.
        - If p-value >= 0.05, the null hypothesis is not rejected.
        """
        if p_value < 0.05:
            result = "Reject the null hypothesis"
        else:
            result = "Fail to reject the null hypothesis"
        
        print(f"{test_name}: p-value = {p_value:.4f} -> {result}")

