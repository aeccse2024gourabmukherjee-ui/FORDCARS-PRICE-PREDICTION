from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained ML pipeline
model = joblib.load("ford_car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from the HTML form
        car_data = {
            "model": request.form["model"],
            "year": int(request.form["year"]),
            "transmission": request.form["transmission"],
            "mileage": float(request.form["mileage"]),
            "fuelType": request.form["fuelType"],
            "tax": float(request.form["tax"]),
            "mpg": float(request.form["mpg"]),
            "engineSize": float(request.form["engineSize"])
        }

        # Convert inputs into a DataFrame
        input_df = pd.DataFrame([car_data])

        # Predict car price
        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction=round(float(prediction), 2)
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            error=f"Please check your inputs: {error}"
        )


if __name__ == "__main__":
    app.run(debug=True)