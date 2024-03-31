# Importing necessary libraries
import os
import sys
from src.exception import CustomException  # Importing CustomException from src.exception module
from src.logger import logging  # Importing logging from src.logger module
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Data class for configuration parameters related to data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Default path for train data
    test_data_path: str = os.path.join('artifacts', 'test.csv')    # Default path for test data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')     # Default path for raw data

# Class for data ingestion process
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initializing data ingestion configuration

    # Method to initiate data ingestion process
    def initiate_data_ingestion(self):
        logging.info('Entered the Data Ingestion method')  # Logging information

        try:
            # Reading the dataset into a DataFrame
            df = pd.read_csv(r'data\data_for_student_performance.csv')
            logging.info('Read the dataset as dataframe')  # Logging information

            # Creating directories if they don't exist for storing data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            logging.info('After retrieving the information from Database, storing it as raw data')  # Logging information
            
            # Saving DataFrame as raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test split initiated')  # Logging information
            
            # Splitting data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Saving train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion Completed')  # Logging information

            # Returning paths of train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )            
        except Exception as e:
            # Handling exceptions and raising custom exception
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()  # Creating an instance of DataIngestion class
    obj.initiate_data_ingestion()  # Initiating the data ingestion process
