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

            # Create a CustomData instance using the first row of the DataFrame
            # data = CustomData(
            #     campaign_id=pred_df['campaign_id'].iloc[0],
            #     sender=pred_df['sender'].iloc[0],
            #     subject_len=pred_df['subject_len'].iloc[0],
            #     body_len=pred_df['body_len'].iloc[0],
            #     mean_paragraph_len=pred_df['mean_paragraph_len'].iloc[0],
            #     day_of_week=pred_df['day_of_week'].iloc[0],
            #     is_weekend=pred_df['is_weekend'].iloc[0],
            #     times_of_day=pred_df['times_of_day'].iloc[0],
            #     category=pred_df['category'].iloc[0],
            #     product=pred_df['product'].iloc[0],
            #     no_of_CTA=pred_df['no_of_CTA'].iloc[0],
            #     mean_CTA_len=pred_df['mean_CTA_len'].iloc[0],
            #     is_image=pred_df['is_image'].iloc[0],
            #     is_personalised=pred_df['is_personalised'].iloc[0],
            #     is_quote=pred_df['is_quote'].iloc[0],
            #     is_timer=pred_df['is_timer'].iloc[0],
            #     is_emoticons=pred_df['is_emoticons'].iloc[0],
            #     is_discount=pred_df['is_discount'].iloc[0],
            #     is_price=pred_df['is_price'].iloc[0],
            #     is_urgency=pred_df['is_urgency'].iloc[0],
            #     target_audience=pred_df['target_audience'].iloc[0]
            # )

            # pred_df = data.get_data_as_data_frame()
            # print(pred_df)

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
