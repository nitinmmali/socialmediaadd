import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
col=['Age', 'EstimatedSalary','Gender']

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction= model.predict(final)
    if prediction==1:
        o1="Will Purchase the add"
    else:
        o1="Will not purchase"

    return render_template('index.html',pred=o1)
if __name__=='__main__':
    app.run(debug=True, port= 2525)
