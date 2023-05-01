from flask import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

ownership_model = pickle.load(open('C:/Users/mukes/OneDrive/Desktop/classes/DSP/lawn_mover.csv', "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def post():
    if request.method == "POST":
        Income = float(request.form["Income"])
        Lot_Size = float(request.form["Lot_Size"])
        df = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})
        result = ownership_model.predict(df)
        probability = ownership_model.predict_proba(df)
        ownership = ('Not owner', 'owner')
        return_str = f"\nThe ownership model indicates probability of cancer at {probability[0][1]:.4f}, therefore it's indicated that we should {ownership[result[0]]}.\n"
        return_str += "<br><a href='/'>Back</a>"
        return return_str

    return render_template("home.html")

if __name__ == "__main__":
    app.run()

