import pickle
import sys
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from  src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException

application=Flask(__name__)

app=application


def read_data_from_csv(file):
    try:
        # Assuming you have a function to read data from a CSV file
        # Modify this accordingly based on your data processing logic
        return pd.read_csv(file)

    except Exception as e:
        raise CustomException(e, sys)
## Route home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and file.filename.endswith('.csv'):
            # Assuming you have a function to read data from a CSV file
            # Modify this accordingly based on your data processing logic
            pred_df = read_data_from_csv(file)

            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = []
            for index, row in pred_df.iterrows():
                    row_df = pd.DataFrame([row])  # Create a DataFrame with the current row
                    result = predict_pipeline.predict(row_df)
                    results.append(result[0])
            
            print("After Prediction")
            return render_template('home.html', results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
