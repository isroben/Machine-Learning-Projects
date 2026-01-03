import sys
import os
from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utility import save_object
from src.components.model_evaluation import evaluate_models

from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Splitting training and testing input data.")

            models = {"Random Forest": RandomForestClassifier(),
                      "Decision Tree": DecisionTreeClassifier(),
                      "Gradient Boosting": GradientBoostingClassifier(),
                      "Logistic Regression": LogisticRegression(),
                      "K-Neighbors Classifier": KNeighborsClassifier(),
                      "Support Vector Machine": SVC(),
                      "XGBClassifier": XGBClassifier(),
                      "Adaboost Classifier": AdaBoostClassifier()
                      }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entrophy'],
                    'splitter': ['best', 'random'],
                    'max_features':['sqrt', 'log2']
                },
                "Random Forest": {
                    'criterion': ['gini', 'entrophy'],
                    'splitter': ['best', 'random'],
                    'max_features':['sqrt', 'log2']
                },
                "Gradient Boosting": {
                    'loss': ['log_loss', 'exponential'],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'squrt', 'log2']
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda name: model_report[name]['R2_test_score'])
            best_model = models[best_model_name]

            logger.info(f"Best model on both training and testing data: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(f"Model saved as: model.pkl")

            return best_model, best_model_name, model_report
        except Exception as e:
            raise CustomException(e, sys)