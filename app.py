from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
le1 = pickle.load(open("model/model_encoder.pkl", "rb"))
le2 = pickle.load(open("model/fuel_encoder.pkl", "rb"))
ct = pickle.load(open("model/column_transformer.pkl", "rb"))
sc = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        
        
        model_name = " " + request.form["model"]
        year = float(request.form["year"])
        transmission = request.form["transmission"]

        mileage_km = float(request.form["mileage"])
        kmpl = float(request.form["mpg"])

        mileage = mileage_km / 1.60934
        mpg = kmpl * 2.35215

        fuel = request.form["fuel"]
        tax = 145
        engine_size = float(request.form["engine_size"])

        model_encoded = le1.transform([model_name])[0]
        fuel_encoded = le2.transform([fuel])[0]


        X = np.array([[
            model_encoded,
            year,
            transmission,
            mileage,
            fuel_encoded,
            tax,
            mpg,
            engine_size
        ]], dtype=object)

        X = ct.transform(X)
        X = sc.transform(X)

        prediction = model.predict(X)[0]
        
        gbp_to_inr = 118.0
        prediction = prediction * gbp_to_inr


    return render_template(
        "index.html",

        prediction=round(float(prediction), 2) if prediction is not None else None,

        form_data=request.form if request.method == "POST" else {},

        model_used="Random Forest Regressor"

    )

    

if __name__ == "__main__":
    app.run(debug=True)
