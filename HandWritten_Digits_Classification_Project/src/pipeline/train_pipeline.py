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
        train_set, test_set = ingestion.initiate_data_ingestion()

        logger.info("Data ingestion completed.")

        # Data transformation
        logger.info("Data transformation process initialized.")
        transformation = DataTransformation()
        X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(train_set, test_set)
        logger.info("Data transformation completed.")

        # Model Training
        logger.info("Model trainer initialized")
        trainer = ModelTrainer()
        best_model, best_model_name, model_report = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        logger.info("Model training comopleted.")
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Performance Report: {model_report[best_model_name]}")

        print("\n================ TRAINING COMPLETED =================")
        print(f"Best Model\t\t: {best_model_name}")

        return best_model, best_model_name, model_report
    
    except Exception as e:
        raise CustomException(e, sys)
    
trainingPipeline()