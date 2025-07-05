import pandas as pd
df = pd.read_csv("data/processed/telco_churn_feature_engineered.csv")
print(df['Churn'].value_counts())