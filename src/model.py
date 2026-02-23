import joblib
import optuna

from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from data import X_train, X_val, X_test, y_train, y_val, y_test
from preprocessing_pipeline import build_preprocessing_pipeline


base_models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
        "Support Vector Regression": SVR(max_iter=-1),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(n_jobs=-1),
        "CatBoosting Regression": CatBoostRegressor(verbose=False),
        "XGBusting Regression": XGBRegressor(),
    }


def objective(trial, model_name):
    if model_name == "Linear Regression":
        params = {}

    elif model_name == "Lasso":
        params = {
            "alpha": trial.suggest_float("alpha", 0.1, 10, step=0.1),
            "tol": trial.suggest_float("tol", 0.001, 1, step=0.001),
        }

    elif model_name == "Ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.1, 10, step=0.1),
            "tol": trial.suggest_float("tol", 0.001, 1, step=0.001),
        }

    elif model_name == "K-Nearest Neighbors":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
            "leaf_size": trial.suggest_int("leaf_size", 10, 1000, step=10),
        }

    elif model_name == "Support Vector Regression":
        params = {
            "degree": trial.suggest_int("degree", 1, 10),
            "tol": trial.suggest_float("tol", 0.001, 1, step=0.001),
            "C": trial.suggest_float("C", 1, 1000, step=0.5),
            "epsilon": trial.suggest_float("epsilon", 0.1, 100, step=0.1),
        }

    elif model_name == "Decision Tree Regression":
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 1000),
        }

    elif model_name == "Random Forest Regression":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=10),
            "max_depth": trial.suggest_int("max_depth", 1, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 1000),
        }

    elif model_name == "CatBoosting Regression":
        params = {
            "iterations": trial.suggest_int("iterations", 1, 501, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5, step=0.05),
            "depth": trial.suggest_int("depth", 1, 10),
        }

    elif model_name == "XGBusting Regression":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5, step=0.05),
            "max_depth": trial.suggest_int("max_depth", 1, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0, 1, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1, step=0.1),
        }

    model = base_models[model_name]
    model.set_params(**params)

    full_pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", model),
    ])

    scores = cross_val_score(
        full_pipeline,
        X_train,
        y_train,
        cv=3,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    return abs(scores.mean())


def tune_models(n_trials):
    models_cv_score = {}
    models_best_params = {}

    for model_name in base_models.keys():
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)

        models_cv_score[model_name] = study.best_value
        models_best_params[model_name] = study.best_params

    return models_cv_score, models_best_params


def fit_and_select_best(models_best_params):
    models_val_score = {}
    fitted_pipelines = {}

    for model_name in base_models.keys():
        model = base_models[model_name]
        params = models_best_params[model_name]
        model.set_params(**params)

        pipeline = Pipeline([
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", model)
            ])

        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        models_val_score[model_name] = root_mean_squared_error(y_val, y_val_pred)
        fitted_pipelines[model_name] = pipeline

    best_model_name = min(models_val_score, key=models_val_score.get)
    return best_model_name, models_val_score[best_model_name], fitted_pipelines[best_model_name]


def test_score(pipeline):
    y_test_pred = pipeline.predict(X_test)
    return root_mean_squared_error(y_test, y_test_pred)

def save_object(object, file_path):
    joblib.dump(object, file_path)