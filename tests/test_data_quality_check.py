import unittest
import pandas as pd
import pandas as pd
import sys
import os
current_dir = os.getcwd()

# Append the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scripts.data_quality_checker import DataQualityChecker 

class TestDataQualityChecker(unittest.TestCase):
    """
    A class to test the data quality of a pandas DataFrame.
    """

    def setUp(self):
        """
        Set up the test data.
        """
        self.data = pd.DataFrame({
            'column1': [1, 2, None, 4],
            'column2': [None, 5, 6, 7],
            'column3': ['a', 'b', 'c', 'd']
        })
        self.expected_types = {
            'column1': 'float64',
            'column2': 'float64',
            'column3': 'object'
        }
        self.checker = DataQualityChecker(self.data)

    def test_check_missing_values(self):
        """
        Test the missing values check function.
        """
        report_df = self.checker.check_missing_values()
        
        # Assert that the report DataFrame contains the correct columns
        self.assertTrue('Column' in report_df.columns)
        self.assertTrue('Missing Count' in report_df.columns)
        self.assertTrue('Missing Percentage' in report_df.columns)

        # Assert specific values
        self.assertEqual(report_df.loc[report_df['Column'] == 'column1', 'Missing Count'].values[0], 1)
        self.assertEqual(report_df.loc[report_df['Column'] == 'column2', 'Missing Percentage'].values[0], 25.0)

    def test_check_data_types(self):
        """
        Test the data types check function.
        """
        report_df = self.checker.check_data_types(self.expected_types)

        # Assert that the report DataFrame contains the correct columns
        self.assertTrue('Column' in report_df.columns)
        self.assertTrue('Actual Data Type' in report_df.columns)
        self.assertTrue('Expected Data Type' in report_df.columns)
        self.assertTrue('Match' in report_df.columns)

        # Assert specific values
        self.assertTrue(report_df.loc[report_df['Column'] == 'column1', 'Match'].values[0])  # Check if 'column1' has the expected type
        self.assertTrue(report_df.loc[report_df['Column'] == 'column2', 'Match'].values[0])  # Check if 'column2' has the expected type



    def test_check_duplicates(self):
        """
        Test the duplicates check function.
        """
        message = self.checker.check_duplicates()
        
        # Assert the message about duplicates
        self.assertEqual(message, "No duplicate rows found in the DataFrame.")

if __name__ == '__main__':
    unittest.main()