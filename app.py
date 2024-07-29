from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                LIMIT_BAL=request.form.get('LIMIT_BAL'),
                SEX=request.form.get('SEX'),
                EDUCATION=request.form.get('EDUCATION'),
                MARRIAGE=request.form.get('MARRIAGE'),
                AGE=request.form.get('AGE'),
                PAY_1=request.form.get('PAY_1'),
                PAY_2=request.form.get('PAY_2'),
                PAY_3=request.form.get('PAY_3'),
                PAY_4=request.form.get('PAY_4'),
                PAY_5=request.form.get('PAY_5'),
                PAY_6=request.form.get('PAY_6'),
                BILL_AMT1=request.form.get('BILL_AMT1'),
                BILL_AMT2=request.form.get('BILL_AMT2'),
                BILL_AMT3=request.form.get('BILL_AMT3'),
                BILL_AMT4=request.form.get('BILL_AMT4'),
                BILL_AMT5=request.form.get('BILL_AMT5'),
                BILL_AMT6=request.form.get('BILL_AMT6'),
                PAY_AMT1=request.form.get('PAY_AMT1'),
                PAY_AMT2=request.form.get('PAY_AMT2'),
                PAY_AMT3=request.form.get('PAY_AMT3'),
                PAY_AMT4=request.form.get('PAY_AMT4'),
                PAY_AMT5=request.form.get('PAY_AMT5'),
                PAY_AMT6=request.form.get('PAY_AMT6')
        )
               
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")   