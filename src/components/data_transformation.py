import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, input_feature_train_df):
        try:
            onehot_encoding_features = ['is_weekend', 'is_personalised', 'is_discount', 'is_price', 'is_urgency', 'day_of_week', 'times_of_day']

            numerical_columns = [col for col in input_feature_train_df.columns if col not in onehot_encoding_features]
            cat_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline,onehot_encoding_features),
                ("num_pipeline",num_pipeline,numerical_columns)
                ]
            )

            logging.info(f"Encoded columns")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


    def handle_outlier(self,df,feature):
        try:
            IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
            lower_limit = df[feature].quantile(0.25) - (IQR * 1.5)
            upper_limit = df[feature].quantile(0.75) + (IQR * 1.5)

            # # Convert to numeric and then to integer
            # df[feature] = pd.to_numeric(df[feature], errors='coerce')
            # df[feature] = df[feature].astype(pd.Int64Dtype())

            # Handle outliers
            df.loc[df[feature] >= upper_limit, feature] = upper_limit
            df.loc[df[feature] <= lower_limit, feature] = lower_limit

            return df[feature]

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and test data")
            logging.info("Preporcoessing object ")

            target_column_name=["click_rate"]

            outlier_features=['no_of_CTA','mean_paragraph_len','body_len','subject_len','mean_CTA_len']

            for feature in outlier_features:
                self.handle_outlier(train_df,feature)
                self.handle_outlier(test_df,feature)



            target_column_name = ["click_rate"]

            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            preprocessing_obj=self.get_data_transformer_object(input_feature_train_df)

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            print(train_arr.shape)
            print(test_arr.shape)
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)            

if __name__=="__main__":
    ob=DataTransformationConfig()
    print(ob.initiate_data_transformation)
