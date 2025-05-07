from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

