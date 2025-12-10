import sys
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

logger = get_logger(__name__)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains multiple models with hyperparameter tuning and returns performance scores.
    """
    try:

        report = {}
        trained_model = {}
        
        for model_name, model in models.items():
            param = params.get(model_name, {})

            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)

            print(f"Best Parameter found by GridSearch:\n{gs.best_params_}")

            # Set best parameters and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            r2_train_score = r2_score(y_train, y_train_pred)
            r2_test_score = r2_score(y_test, y_test_pred)

            # # Calculate Mean Squared Error
            # mse_train_score = mean_squared_error(y_train, y_train_pred)
            # mse_test_score = mean_squared_error(y_test, y_test_pred)

            # # Calculating Mean Absolute Error
            # mae_train_score = mean_absolute_error(y_train, y_train_pred)
            # mae_test_score = mean_absolute_error(y_test, y_test_pred)

            print(f"R2 Train Score:\n{r2_train_score}")
            print(f"R2 Test Score:\n{r2_test_score}")
            
            report[model_name] = {
                'R2_train_score': r2_train_score,
                'R2_test_score': r2_test_score,
                'best_params': gs.best_params_
            }
            logger.info(f"Report Stored!")

            # Store trained model
            trained_model[model_name] = model
            logger.info(f"trained model Stored!")


        return report, trained_model
        
    
    except Exception as e:
        raise CustomException(e, sys)