import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/telco_churn.csv")
    
    # Create tenure groups
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    
    # Convert 'Churn' to numeric
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Save feature engineered data
    df.to_csv("../../data/processed/telco_churn_feature_engineered.csv", index=False)