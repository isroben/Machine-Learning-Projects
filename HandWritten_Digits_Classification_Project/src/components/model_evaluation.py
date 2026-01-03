import sys
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

logger = get_logger(__name__)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        trained_model = {}

        for model_name, model in models.items():
            param = param.get(model_name, {})

            gs = GridSearchCV(model, param, cv=10)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            cnMtrx_train = confusion_matrix(y_train, y_train_pred)
            cnMtrx_test = confusion_matrix(y_test, y_test_pred)

            preScr_train = precision_score(y_train, y_train_pred)
            preScr_test = precision_score(y_test, y_test_pred)

            accScr_train = accuracy_score(y_train, y_train_pred)
            accScr_test = accuracy_score(y_test, y_test_pred)

            f1Scr_train = f1_score(y_train, y_train_pred)
            f1Scr_test = f1_score(y_test, y_train_pred)

            print(f"Confussion metrics: {cnMtrx_train}")
            print(f"Confussion metrics: {cnMtrx_test}")

            print(f"\nConfussion metrics: {preScr_train}")
            print(f"Confussion metrics: {preScr_test}")

            print(f"\nConfussion metrics: {accScr_train}")
            print(f"Confussion metrics: {accScr_test}")

            print(f"\nConfussion metrics: {f1Scr_train}")
            print(f"Confussion metrics: {f1Scr_test}")

            report[model_name] = {
                'Confussion Metrix train': cnMtrx_train,
                'Confussion Metrix test': cnMtrx_test,
                'best_params': gs.best_params_
            }

            logger.info(f"Report Stored!")

            trained_model[model_name] = model
            logger.info("trained model stored.")

        return report, trained_model
        
    except Exception as e:
        raise CustomException(e, sys)
            