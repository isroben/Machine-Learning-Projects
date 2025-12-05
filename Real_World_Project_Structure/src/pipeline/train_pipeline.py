import sys
import os
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

logger = get_logger(__name__)

def training_pipeline():
    try:
        logger.info("Training Pipeline started.")

        # Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        logger.info("Data ingetion completed.")

        # Data Transformation
        transform = DataTransformation()
        train_array, test_array, _ = transform.initiate_data_transformation(train_data_path, test_data_path)

        logger.info("Data transformation completed.")

        # Model Training
        trainer = ModelTrainer()
        best_model, best_model_name, model_report = trainer.intiate_model_trainer(train_array, test_array)

        logger.info("Model training completed.")
        logger.info(f"Best Moel Selected: {best_model_name}")
        logger.info(f"Performance Report: {model_report[best_model_name]}")


        print("\n================ TRAINING COMPLETED =================")
        print(f"Best Model          : {best_model_name}")
        print(f"Best Model Train R2 : {model_report[best_model_name]['R2_train_score']}")
        print(f"Best Model Test R2  : {model_report[best_model_name]['R2_test_score']}")
        print("Model saved at: artifacts/model.pkl")
        print("=====================================================\n")

        return best_model, best_model_name, model_report
    
    except Exception as e:
        raise CustomException(e, sys)
    
training_pipeline()