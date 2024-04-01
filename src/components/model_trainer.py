import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        '''
        This function is used to initialise the model training on the training data and save the model.
        '''

        try:
            logging.info('Split the training and testing data')
            X_train= train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            models = {
                'Random Forest': RandomForestRegressor(verbose=False),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(verbose=False),
                'Linear Regressor': LinearRegression(),
                'XGB Regressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'ADABoost Regressor': AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            # best_model_name = list(model_report.keys)[list(model_report.value().index(best_model_score))]
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best Model found')
            
            logging.info(f'Best model found and it is {best_model_name}')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            y_predicted = best_model.predict(X_test)
            r2_value = r2_score(y_test, y_predicted)

            return r2_value

        except Exception as e:
            CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_data_arr, test_data_arr, __ = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)

    modeltrainer = ModelTrainer()
    r2_score = modeltrainer.initiate_model_training(train_array= train_data_arr, test_array= test_data_arr)

    print(r2_score)