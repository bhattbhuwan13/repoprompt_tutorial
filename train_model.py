import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Read the data
    data_path = os.path.join(os.path.dirname(__file__), "data", "WineQT.csv")
    df = pd.read_csv(data_path)

    # Prepare features and target
    X = df.drop(columns=["quality", "Id"])
    y = df["quality"]

    # Split data (not strictly necessary for training, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "decision_tree_wineqt.joblib")
    joblib.dump(clf, model_path)

    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    main()