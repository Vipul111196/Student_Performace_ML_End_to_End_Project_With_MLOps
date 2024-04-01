import os
import sys
import pandas as pd
import numpy as np
import dill #dill is a library similar to pickle but can serialize a wider range of Python objects.
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        '''
        dill.dump(obj, file_obj): This line uses the dill.dump() 
        function to serialize the Python object obj and 
        write it to the file object file_obj. 
        
        dill is a library similar to pickle but can serialize a wider range of Python objects.
        '''

    except Exception as e:
        raise CustomException(e, sys)
