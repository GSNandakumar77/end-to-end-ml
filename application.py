from flask import Flask, request, render_template
import os
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open("models/ridge_model.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            scaled_data = standard_scaler.transform(input_data)
            prediction = ridge_model.predict(scaled_data)[0]

            return render_template("prediction.html", result=prediction)
        except Exception as e:
            return render_template("prediction.html", result=f"Error: {str(e)}")
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
