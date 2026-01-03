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

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logger.info("Splitting training and testing input data.")

            X_train, y_train, X_test, y_test = ()
        except:
            pass