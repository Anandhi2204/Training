from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model_FRESH.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    temp = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Prepare data
    features = np.array([[temp, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction_text=f"Recommended Crop: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
