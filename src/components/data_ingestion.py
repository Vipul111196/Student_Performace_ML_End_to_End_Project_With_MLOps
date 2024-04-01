# Importing necessary libraries
import os
import sys
from src.exception import CustomException  # Importing CustomException from src.exception module
from src.logger import logging  # Importing logging from src.logger module
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Data class for configuration parameters related to data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Default path for train data
    test_data_path: str = os.path.join('artifacts', 'test.csv')    # Default path for test data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')     # Default path for raw data

'''This code defines a Python class named DataIngestionConfig using the @dataclass decorator. 
The @dataclass decorator is a feature introduced in Python 3.7 that automatically generates special methods 
like __init__, __repr__, __eq__, and __hash__ based on class variables.'''

'''By using the @dataclass decorator, Python automatically generates special methods for the class based on its attributes. 
This makes the class easier to use and understand, as you don't need to manually implement these methods.'''

# Class for data ingestion process
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initializing data ingestion configuration

    '''__init__(self): This is the constructor method of the DataIngestion class. 
    It gets called when a new instance of DataIngestion is created. 
    Inside this constructor, an instance of DataIngestionConfig is created.'''

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