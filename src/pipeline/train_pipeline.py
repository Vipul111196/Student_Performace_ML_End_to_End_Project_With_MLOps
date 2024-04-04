from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

# Initialising the Data ingestion
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()
train_data_arr, test_data_arr, __ = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)

modeltrainer = ModelTrainer()
best_model_name, best_model_score = modeltrainer.initiate_model_training(train_array= train_data_arr, test_array= test_data_arr)

logging.info(f'Out of 7 model training, the best model is {best_model_name} and its R2 score is {best_model_score}')
print(f'Out of 7 model training, the best model is {best_model_name} and its R2 score is {best_model_score}')