import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.exception import CustomException
from utils.logger import get_logger
from utils.utils import save_object

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        Returns a ColumnTransformer for preprocessing.
        """
        try:
            numerical_columns = []
            categorical_columns = []

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns)
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path, target_column=''):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logger.info("Train/Test data loaded successfully.")
            logger.info("Creating preprocessing object...")

            preprocessor = self.get_data_transformer_obj()

            X_train = train.drop(target_column, axis=1)
            y_train = train[target_column]

            X_test = test.drop(target_column, axis=1)
            y_test = test[target_column]

            logger.info(f"Applying preprocessing object on train and test data.")

            X_train_array = preprocessor.fit_transform(X_train)
            X_test_array = preprocessor.fit_transform(X_test)

            logger.info(f"Preprocessing completed. Training shape: {X_train_array.shape}")

            train_array = np.column_stack((X_train_array, y_train.to_numpy()))
            test_array = np.column_stack((X_test_array, y_test.to_numpy()))

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logger.info("Preprocessing saved successfully!")

            return train_array, test_array, self.config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
            