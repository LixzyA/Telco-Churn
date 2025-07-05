import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_churn_feature_engineered.csv")

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Define features and target
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the model
    joblib.dump(model, 'models/randomforest_model.joblib')

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Precision: 0.77
    # Recall: 0.78
    # f1-score: 0.77