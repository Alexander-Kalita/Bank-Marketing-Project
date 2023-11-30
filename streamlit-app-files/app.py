import joblib
import json
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_cv.json")

expected_number_of_features = 42
def predict_single(features, model):
    feature_names = model.get_booster().feature_names
    features = [features.get(feature, 0) for feature in feature_names]
    X = np.array(features).reshape(1, -1)
    prediction = loaded_model.predict_proba(X)
    return float(prediction[0, 1])
app = Flask('banksubscriptions')
@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
            features = data.get('features', {})
        else:
            features = request.args.get('features')
            features = json.loads(features) if features else {}
        if len(features) != expected_number_of_features:
            return jsonify({'error': 'Invalid number of features'}), 400
        prediction = predict_single(features, loaded_model)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
