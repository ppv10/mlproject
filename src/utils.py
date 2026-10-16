import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, models, param):
    """Fit each candidate and return its training-only CV score for model selection."""
    try:
        report = {}

        for name, model in models.items():
            search_space = param.get(name, {})
            if search_space:
                search = RandomizedSearchCV(
                    model,
                    search_space,
                    cv=5,
                    n_jobs=-1,
                    random_state=0,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                model.set_params(**search.best_params_)
                cv_score = search.best_score_
            else:
                cv_score = cross_val_score(
                    model, X_train, y_train, cv=5, n_jobs=-1, scoring="r2"
                ).mean()

            # Fit the selected configuration on all training rows. The test set is
            # reserved for the single final evaluation in ModelTrainer.
            model.fit(X_train, y_train)
            report[name] = float(cv_score)

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)