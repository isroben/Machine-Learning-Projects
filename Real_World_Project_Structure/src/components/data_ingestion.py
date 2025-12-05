import os
import sys

from utils.exception import CustomException
from utils.logger import get_logger

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join('data', 'income.csv')
    train_data_path: str = os.path.join('data', 'train.csv')
    test_data_path: str = os.path.join('data', 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process.")

        try:
            # Validate file
            if not os.path.exists(self.config.source_data_path):
                raise CustomException(f"Source data file does not exist: {self.config.source_data_path}", sys)
            
            # Load dataset
            dataset = pd.read_csv(self.config.source_data_path)
            logger.info("Dataset loaded successfully.")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw dataset
            dataset.to_csv(self.config.raw_data_path, index=False)
            logger.info("Raw dataset saved.")

            # Train-test split
            train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
            logger.info("Train-test split completed.")

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info("Data Ingestin completed.")

            return self.config.train_data_path, self.config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
