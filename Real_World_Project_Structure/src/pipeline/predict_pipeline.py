import sys
import pandas as pd
from src.utils.exception import CustomException
from src.utils.utility import load_object


class PredictPipeline:
    def __init__(self):
        try:
            self.preprocessor = load_object(r'')
            self.model = load_object(r'')
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            print(f'Data scaled after transformation is: \n {data_scaled}')
            preds = self.model.predict(data_scaled)


            return preds
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.data]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        