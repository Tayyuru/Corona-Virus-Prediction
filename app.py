import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    columns = ['Age',	'Body Temperature',	'Fatigue',	'Cough',	'Body Pains',	'SoreThroat',	'Breathing Difficulty']
    
    float_features = [float(request.form.get(col)) for col in columns]

    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction==1:
        return render_template("index.html",prediction_text='You have Corona Virus Symptoms, Get Treatment')
    else:
        return render_template("index.html",prediction_text='You have no Corona Virus Symptoms, You are Safe')

if __name__ == "__main__":
    flask_app.run(debug=True)
