import os
import sys

from src.utils.exception import CustomException
from src.utils.logger import get_logger

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from dataclasses import dataclass


logger = get_logger(__name__)

try:
    digits = load_digits()
    dataset = pd.DataFrame(digits.data)
    y = digits.target

    dataset['target'] = y

except Exception as e:
    raise CustomException(e, sys)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'train.csv')
    test_data_path: str = os.path.join('data', 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logger.info("Starting data ingestion process.")

        try:
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            dataset.to_csv(self.config.raw_data_path, index=False)
            logger.info("Raw Dataset saved.")

            # Train test split
            train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
            logger.info("Train test split completed.")

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info("Data Ingestion completed.")

            return train_set, test_set
            
        except Exception as e:
            raise CustomException(e, sys)

