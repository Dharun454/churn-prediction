from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model_path = 'pretrained_model.joblib'
pipeline = joblib.load(model_path)

# Home route to serve the front-end HTML form
@app.route('/')
def home():
    return render_template('index.html')

# API route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    form_data = request.form.to_dict()
    
    # Convert form data to a DataFrame (one row)
    df = pd.DataFrame([form_data])
    
    # Ensure correct data types for the model (e.g., numeric values)
    df['rownumber'] = pd.to_numeric(df['rownumber'])
    df['creditscore'] = pd.to_numeric(df['creditscore'])
    df['age'] = pd.to_numeric(df['age'])
    df['tenure'] = pd.to_numeric(df['tenure'])
    df['balance'] = pd.to_numeric(df['balance'])
    df['numofproducts'] = pd.to_numeric(df['numofproducts'])
    df['hascrcard'] = pd.to_numeric(df['hascrcard'])
    df['isactivemember'] = pd.to_numeric(df['isactivemember'])
    df['estimatedsalary'] = pd.to_numeric(df['estimatedsalary'])

    # Make prediction
    prediction = pipeline.predict(df)[0]
    
    # Return prediction as a JSON response
    return jsonify({'prediction': 'Churn' if prediction == 1 else 'No Churn'})

if __name__ == '__main__':
    app.run(debug=True)
