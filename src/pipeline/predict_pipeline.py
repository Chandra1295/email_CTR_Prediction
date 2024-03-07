import sys
import pickle
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 campaign_id: str,
                 sender: str,
                 subject_len: int,
                 body_len: int,
                 mean_paragraph_len: float,
                 day_of_week: str,
                 is_weekend: int,
                 times_of_day: str,
                 category: str,
                 product: str,
                 no_of_CTA: int,
                 mean_CTA_len: float,
                 is_image: int,
                 is_personalised: int,
                 is_quote: int,
                 is_timer: int,
                 is_emoticons: int,
                 is_discount: int,
                 is_price: int,
                 is_urgency: int,
                 target_audience: str):
        self.campaign_id = campaign_id
        self.sender = sender
        self.subject_len = subject_len
        self.body_len = body_len
        self.mean_paragraph_len = mean_paragraph_len
        self.day_of_week = day_of_week
        self.is_weekend = is_weekend
        self.times_of_day = times_of_day
        self.category = category
        self.product = product
        self.no_of_CTA = no_of_CTA
        self.mean_CTA_len = mean_CTA_len
        self.is_image = is_image
        self.is_personalised = is_personalised
        self.is_quote = is_quote
        self.is_timer = is_timer
        self.is_emoticons = is_emoticons
        self.is_discount = is_discount
        self.is_price = is_price
        self.is_urgency = is_urgency
        self.target_audience = target_audience

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "campaign_id": [self.campaign_id],
                "sender": [self.sender],
                "subject_len": [self.subject_len],
                "body_len": [self.body_len],
                "mean_paragraph_len": [self.mean_paragraph_len],
                "day_of_week": [self.day_of_week],
                "is_weekend": [self.is_weekend],
                "times_of_day": [self.times_of_day],
                "category": [self.category],
                "product": [self.product],
                "no_of_CTA": [self.no_of_CTA],
                "mean_CTA_len": [self.mean_CTA_len],
                "is_image": [self.is_image],
                "is_personalised": [self.is_personalised],
                "is_quote": [self.is_quote],
                "is_timer": [self.is_timer],
                "is_emoticons": [self.is_emoticons],
                "is_discount": [self.is_discount],
                "is_price": [self.is_price],
                "is_urgency": [self.is_urgency],
                "target_audience": [self.target_audience],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)