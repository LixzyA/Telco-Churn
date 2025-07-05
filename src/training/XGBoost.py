import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_churn_feature_engineered.csv")

    # Remove the tenure_group feature
    df = df.drop(columns=[col for col in df.columns if 'tenure_group' in col])

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Define features and target
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Initialize the model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='precision', n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the best model
    joblib.dump(best_model, 'models/xgboost_model.joblib')

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
