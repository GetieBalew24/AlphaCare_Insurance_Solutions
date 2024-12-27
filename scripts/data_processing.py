import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def feature_engineering(self):
        # Convert 'VehicleIntroDate' to datetime, handling errors gracefully
        self.data['VehicleIntroDate'] = pd.to_datetime(self.data['VehicleIntroDate'], errors='coerce')

        # Optionally handle rows with invalid or missing 'VehicleIntroDate'
        # Option 1: Drop rows with missing 'VehicleIntroDate'
        self.data = self.data.dropna(subset=['VehicleIntroDate'])

        # Option 2: Or fill missing 'VehicleIntroDate' with the current date (if you prefer)
        # self.data['VehicleIntroDate'] = self.data['VehicleIntroDate'].fillna(pd.Timestamp.now())

        # Calculate 'VehicleAge' in years (based on the current date)
        self.data['VehicleAge'] = (pd.Timestamp.now() - self.data['VehicleIntroDate']).dt.days // 365

        # Continue with other feature engineering steps as required
        # For example, creating additional features, encoding, etc.
        # self.data['some_feature'] = ...

        return self.data
    def handle_missing_values(self):
        # Handle missing values logic...
        for column in self.data.columns:
            if self.data[column].dtype in ['float64', 'int64']:
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif self.data[column].dtype == 'object':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        return self.data

    def encode_categorical_data(self, method, columns):
        # Encode categorical data logic...
        for col in columns:
            if method == 'onehot':
                self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
            elif method == 'label':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
        return self.data

    def scale_data(self, method='standard'):      
        # Select only numeric columns for scaling
        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling only to numeric columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self.data


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline")
    parser.add_argument('--stage', required=True, help="Stage of the pipeline")
    parser.add_argument('--input', required=True, help="Input file path")
    parser.add_argument('--output', required=True, help="Output file path")

    args = parser.parse_args()

    # Read the data
    data = pd.read_csv(args.input)

    # Initialize the data preprocessing class
    dp = DataPreprocessing(data)

    # Execute the required stage
    if args.stage == 'preprocess':
        data = dp.feature_engineering()
    elif args.stage == 'handle_missing_values':
        data = dp.handle_missing_values()
    elif args.stage == 'encode_data':
        data = dp.encode_categorical_data(method='label', columns=['NewVehicle'])
    elif args.stage == 'scale_data':
        data = dp.scale_data(method='standard')
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    # Save the output data
    data.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
