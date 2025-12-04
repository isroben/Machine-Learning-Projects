import os
import sys
from dataclasses import dataclass

from utils.exception import CustomException
from utils.logger import get_logger
from utils.utils import save_object, load_object
from components.model_evaluation import evaluate_models

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

# from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def intiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and testing input data.")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                'Support Vector Machine': SVR(gamma='auto'),
                "XGBRegressor": XGBRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },

                'Random Forest': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', 'None'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'Gradient Boosting': {
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'squrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'Linear Regression':{},
                'XGBRegressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },

                'Support Vector machine': {
                    'C':[0.1,0.3,0.5,1,10,20],
                    'kernel':['rbf', 'linear']
                },

                'AdaBoost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Pick the best Model
            best_model_name = max(model_report, key=lambda name: model_report[name]['R2_test_score'])
            best_model = models[best_model_name]

            logger.info(f"Best model on both training and testing data: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(f"Model saved as: model.pkl")
        except Exception as e:
            raise CustomException(e, sys)



