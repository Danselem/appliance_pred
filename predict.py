import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_lr=0.1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('energy')

@app.route('/predict', methods=['POST'])
def predict():
    appliance = request.get_json()

    X = dv.transform([appliance])
    energy = model.predict(X)
    prediction = np.exp(energy)

    result = {
        'prediction': float(energy),
        'value': float(prediction)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)