import sys
import pandas as pd
from src.utils.exception import CustomException
from src.utils.utility import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = r''
            preprocessor_path = r''
            print("Before loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Print(f"Features before transformation is:\n {features}")
            data_scaled = preprocessor.transform(features)
            print(f'Data scaled after transformation is: \n {data_scaled}')
            preds = model.predict(data_scaled)


            return preds
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, age:int):
        self.age = age

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        