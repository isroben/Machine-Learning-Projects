import sys
import os
from dataclasses import dataclass
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utility import save_object

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('numerical_pipeline', numerical_pipeline, slice(0, -1))
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_set, test_set, target_column):
        try:

            logger.info("Train/Test data loaded successfully.")
            logger.info("Creating proprocessing object...")

            preprocessor = self.get_data_transformer_obj()

            X_train = train_set.drop(column=target_column, axis=1)
            y_train = train_set[target_column]
            X_test = test_set.drop(column=target_column, axis=1)
            y_test = train_set[target_column]

            logger.info(f"Applying preprocessing object on train and test data.")

            X_train_array = preprocessor.fit_transform(X_train)
            X_test_array = preprocessor.fit_transform(X_test)

            logger.info(f"Preprocessing completed. Training shape: {X_train_array.shape}")