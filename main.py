import pandas as pd
import pickle
import numpy as np
from flask import Flask,render_template,request

app=Flask(__name__)
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("Ridgemodel.pkl",'rb'))


@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])

def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')
    
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0] * 1e5
    if prediction >= 1e7:  # If prediction is greater than or equal to 1 crore
        prediction_in_crores = prediction / 1e7
        return f'{np.round(prediction_in_crores, 2)} Crores'
    else:
        prediction_in_lakhs = prediction / 1e5
        return f'{np.round(prediction_in_lakhs, 2)} Lakhs'
    


if __name__=="__main__":
    app.run(debug=True,port=5001)