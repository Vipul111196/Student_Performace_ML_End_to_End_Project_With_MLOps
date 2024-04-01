import os 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        
        This function is responsible for data transformation.

        '''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                                'test_preparation_course']
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            '''
            num_pipeline:

            The pipeline consists of two steps:
            - "imputer": It uses SimpleImputer to fill missing values in numerical features. 
            The strategy used here is to replace missing values with the mean of the column.

            - "scaler": It scales the features using StandardScaler. 
            This step standardizes the features by removing the mean and scaling to unit variance. 
            with_mean=False indicates that it centers the data before scaling by subtracting the mean.'''
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            '''
            categorical_pipeline:

            The pipeline also consists of three steps:
            - "imputer": It uses SimpleImputer to fill missing values in categorical features. 
            The strategy used here is to replace missing values with the most frequent value of the column.
            - "one_hot_encoder": It performs one-hot encoding using OneHotEncoder. 
            One-hot encoding converts categorical variables into binary vectors where each category is represented as a binary feature.
            - "scaler": Similar to num_pipeline, it scales the features using StandardScaler with with_mean=False to center the data before scaling.'''
            
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            # preprocessor = ColumnTransformer([
            #     ('num_pipeline', num_pipeline, numerical_columns),
            #     ('categorical_pipeline', categorial_pipeline, categorial_columns)  
            # ])
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            '''
            ColumnTransformer:

            The ColumnTransformer is created to apply different transformations to different columns of the input data.
            It takes a list of tuples where each tuple consists of:
            - A name: This is a string identifier for the transformation.
            - A transformer: This is the pipeline or transformer to apply to the specified columns.
            - Columns: This specifies the columns to apply the transformer to.

            The ColumnTransformer is initialized with two transformations:

            - "num_pipeline": This applies the num_pipeline to the columns specified by numerical_columns. 
            This pipeline is designed for numerical features and includes steps for imputation and scaling.
            -"categorical_pipeline": This applies the categorical_pipeline to the columns specified by categorical_columns. 
            This pipeline is designed for categorical features and includes steps for imputation, one-hot encoding, and scaling.
            '''
            
            logging.info('Categorical Features Created using One hot encoder')
            logging.info('Numerical Columns scaling completed')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)

            logging.info(f'This is the training data 1 row ')
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Preprocessing of train and test data started')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns= target_column_name, axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            '''
            This line applies the preprocessing pipeline to the training and testing dataset input features, 
            transforming them into an array suitable for training machine learning models. 
            
            The fit_transform() method is used here, which fits the transformers to the training and testing data and 
            then transforms it.
            '''

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            '''
            These lines concatenate the transformed input features with the corresponding target features. 
            This likely creates the final arrays that will be used for training and testing machine learning models.
            '''

            logging.info(f'Saved preprocessing object.')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info(f"Saved preprocessing object.")

            '''
            this line of code is essentially saving the preprocessing pipeline object (preprocessing_obj) 
            to a file at the location specified by preprocessor_obj_file_path. 
            This allows the preprocessing steps to be saved and loaded later for reuse, 
            which can be useful in machine learning workflows where preprocessing steps 
            need to be applied consistently to new data.
            '''

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)