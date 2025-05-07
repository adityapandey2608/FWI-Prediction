import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

## Import Ridge regressor model and standard scaler
ridge_model = pickle.load(open('models/regressor.pkl' , 'rb'))
Standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))


## Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata' , methods=['GET','POST'])
def predictdata():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))

        new_data_scaled = Standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result = result[0])
    else:
        return render_template('home.html')




if(__name__) == "__main__":
    app.run(host="0.0.0.0")