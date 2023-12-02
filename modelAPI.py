from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import joblib

app = Flask(__name__)

# Load the pre-trained Isolation Forest model
model = joblib.load('isolation_forest_model.pkl')  # Change the filename as per your saved model

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()  # Get JSON data from POST request
        transaction_data = pd.DataFrame(data, index=[0])

        # Select relevant features for fraud detection
        features = ['block_slot', 'fee', 'compute_units_consumed']
        X = transaction_data[features]

        # Impute missing values (replace NaN with mean of each column)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Predict outliers/anomalies (potentially fraud transactions)
        predictions = model.predict(X_imputed)
        transaction_data['is_fraud'] = predictions

        # Prepare response
        result = {
            'is_fraud': int(predictions[0])  # Convert prediction to integer for response
        }

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the app
