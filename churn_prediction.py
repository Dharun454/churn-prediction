import os
import pandas as pd
import joblib  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (ensure to provide the correct path)
data = pd.read_csv('dataset.csv')  # Change this to your data file path

# Define features and target variable
X = data.drop('churn', axis=1)  # Replace 'churn' with your target variable name
y = data['churn']  # Replace 'churn' with your target variable name

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Path to save or load the trained model
model_path = 'pretrained_model.joblib'

# Check if a model already exists
if os.path.exists(model_path):
    print("Existing model found. Loading the model...")
    pipeline = joblib.load(model_path)
else:
    print("No existing model found. Training a new model...")

# Number of epochs for training
epochs = 1
train_accuracies = []

# Train the model for a specified number of epochs
for epoch in range(epochs):
    # Fit the model
    pipeline.fit(X_train, y_train)
    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    train_accuracies.append(train_accuracy)
    
    print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy:.4f}")

    # Save or update the model after training
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

def test_model(model, X_test, y_test):
    """
    Test the given model on the test data.

    Parameters:
    - model: The trained model (pipeline).
    - X_test: Test features.
    - y_test: True labels for the test features.

    Returns:
    - accuracy: Accuracy of the model on the test set.
    - report: Classification report of the model on the test set.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

# Test the model after training
test_accuracy, report = test_model(pipeline, X_test, y_test)
print("\nFinal Testing Accuracy:", test_accuracy)
print("Classification Report:\n", report)
