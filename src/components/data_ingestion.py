import os
import sys
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging
import pandas as pd

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train.csv")
    test_data_path=os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entering the data")
        
        try:
            train_df=pd.read_csv('notebook/data/train_F3fUq2S.csv')
            logging.info('Exported the train data')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info('Read train data ')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_df,test_df= train_test_split(train_df,test_size=0.2,random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Train test loaded")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ =='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    a=modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(f'{a} cbjdbebcbc')