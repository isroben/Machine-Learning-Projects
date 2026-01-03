import os
import sys
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logger = get_logger(__name__)

def trainingPipeline():
    try:
        logger.info("Training pipeline initialized.")

        # Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.inititate_data_ingestion()

        logger.info("Data ingestion completed.")

        # Data transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model Training
        trainer = ModelTrainer()
        best_model, best_model_name, model_report = trainer.initiate_model_trainer(train_arr, test_arr)

        logger.info("Data transformation completed.")

    except Exception as e:
        raise CustomException(e, sys)