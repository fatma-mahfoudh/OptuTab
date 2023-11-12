import mlflow
mlflow.autolog(disable=True)
import pandas as pd
import optuna
from utils.utils_eval import add_score, plot_results, log_results
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation

def instantiate_xgboost(trial, random_state):
    """Create XGBoost model to tune

    Args:
        trial (Trial): Trial Optuna object
    """
    params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int('n_estimators', 3, 50), #100
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True), #0.1
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), #0.01
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 0.1),
            "random_state": random_state,
            "max_leaves": trial.suggest_int("max_leaves", 6, 20),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            # "early_stopping_rounds":50
        }
    return(XGBRegressor(**params))

def instantiate_elasticnet(trial, random_state):
    """Create ElasticNet model to tune

    Args:
        trial (Trial): Trial Optuna object
    """
    params = {
        "alpha": trial.suggest_float("alpha", 1e-2, 10),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 1),
        "random_state": random_state
    }
    return(ElasticNet(**params))

def instantiate_lgbm(trial, random_state):
    """Create LGBM model to tune

    Args:
        trial (Trial): Trial Optuna object
    """
    params = {
        'metric': 'mae', 
        'random_state': random_state,
        'n_estimators': trial.suggest_int('n_estimators', 3, 30), #100
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 6),
        'num_leaves' : trial.suggest_int('num_leaves', 6, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
    }
    return(LGBMRegressor(**params))

def instantiate_model(trial, model_name, random_state):
    """Create model instance to tune

    Args:
        trial (Trial): Trial Optuna object
        model_name (string): model name

    Raises:
        Exception: exception if model name not in [XGBoost]
    """
    if model_name == 'XGBRegressor':
        model = instantiate_xgboost(trial, random_state)
    elif model_name == 'ElasticNet':
        model = instantiate_elasticnet(trial, random_state)
    elif model_name == 'LGBMRegressor':
        model = instantiate_lgbm(trial, random_state)
    else:
        raise Exception('model name should be in [XGBoost, ElasticNet]')
    return(model)

def objective(trial, model_name, X, y, n_splits, random_state):
    """Define objective function for hyperparameter optimization

    Args:
        trial (Trial): Trial Optuna Object
        model_name (string): model name
        X (DataFrame): Training Features
        y (DataFrame or Series): Training Targets
        n_splits (int): number of splits for cross validation
        random_state (int): random seed
    """
    model = instantiate_model(trial, model_name, random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_scorer = make_scorer(mean_squared_error)
    scores = cross_val_score(model, X, y, scoring=mse_scorer, cv=kf)
    return(np.min([np.mean(scores), np.median(scores)]))

# def tune_lgbm(X, y, n_estimators, random_state):
#     """Tune LGBM model with step wise method

#     Args:
#         X (DataFrame): training features
#         y (Series): training target
#         target_name (string): target name
#         n_estimators (int): number of estimators in LGBM model
#     """
#     dtrain = lgb.Dataset(X, label=y)

#     params = {
#         "objective": "regression",
#         "metric": "l1",
#         "boosting_type": "gbdt",
#         # "n_estimators":n_estimators,
#     }

#     study_tuner = optuna.create_study(direction='minimize')

#     tuner = lgb.LightGBMTunerCV(
#         params,
#         dtrain,
#         folds=KFold(n_splits=3),
#         callbacks=[early_stopping(100)], #log_evaluation(100)
#         return_cvbooster=True,
#         study=study_tuner,
#         time_budget=10*60,
#         optuna_seed=random_state,
#         verbosity=-1,
#         seed=random_state,
#     )
#     optuna.logging.set_verbosity(optuna.logging.ERROR)
#     tuner.run()
#     print(tuner.best_params)
#     return(tuner)

def inverse_transform(df, target2transform, targets):
    """Inverse transform target

    Args:
        df (DataFrame): DataFrame of targets
        target2transform (dictionary): dictionary used for transformation
        targets (list): list of target names
    """
    df_transform = df.copy()
    for target in targets:
        if target2transform[target] == 'log':
            df_transform[target] = np.exp(df_transform[target])
        elif target2transform[target] == '1/sqrt':
            df_transform[target] = 1 / (df_transform[target]) ** 2 + 0.5
        elif target2transform[target] == '1/':
            df_transform[target] = 1 / df_transform[target]
        elif target2transform[target] == '':
            continue
    return(df_transform)
 
def tune_model(run_name, model_name, n_trials, targets, X_train, y_train, X_test, y_test, n_splits, 
               random_state, mlflow_model_name, score_dict, transform_target=False, target2transform=None):
    """Tune model and plot results

    Args:
        run_name (string): mlflow run name
        model_name (string): model name
        n_trials (int): number of trials
        targets (list): list of targets
        X_train (DataFrame): training features
        y_train (DataFrame): training targets
        X_test (DataFrame): testing features
        y_test (DataFrame): testing targets
        n_splits (int): number of splits for cross validation
        random_state (int): number to fix seed 
        mlflow_model_name (string): model name to be used for mlflow logging
        score_dict (dictionary): dictionary score
        transform_target (boolean): set to true if target transformation is needed
        target2transform (dict): dictionary of target:transformation
    """
    if model_name not in ['XGBRegressor', 'ElasticNet', 'LGBMRegressor']:
        raise Exception('model name should be in [XGBRegressor, ElasticNet, LGBMRegressor]')
    model = getattr(sys.modules[__name__], model_name)
    with mlflow.start_run(run_name=run_name):
        y_pred_train = pd.DataFrame([])
        y_pred_test = pd.DataFrame([])
        opt_model_dict = {}
        for target in targets:
            print(f'Started tuning {model_name} for {target}...')
            # if model_name != 'LGBMRegressor':
            study = optuna.create_study(direction="minimize", study_name="regression", 
                                        sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True))
            study.optimize(lambda trial: objective(trial, model_name, X_train, y_train[target], n_splits, random_state), 
                        n_trials=n_trials)
            # else:
            #     study = tune_lgbm(X_train, y_train[target], n_estimators=5, random_state=random_state)
            print(f'Tuning finished! \n')
            opt_model = model(**study.best_params)
            opt_model.fit(X_train, y_train[target])
            y_pred_train_target = pd.DataFrame(opt_model.predict(X_train), columns=[target])
            y_pred_test_target = pd.DataFrame(opt_model.predict(X_test), columns=[target])
            opt_model_dict[target] = opt_model
            y_pred_train = pd.concat([y_pred_train, y_pred_train_target], axis=1)
            y_pred_test = pd.concat([y_pred_test, y_pred_test_target], axis=1)
        if transform_target:
            y_pred_train = inverse_transform(y_pred_train, target2transform, targets)
            y_pred_test = inverse_transform(y_pred_test, target2transform, targets)
            y_train = inverse_transform(y_train, target2transform, targets)
            y_test = inverse_transform(y_test, target2transform, targets)
        score_dict, y_model_dict = log_results(y_pred_train, y_train, y_pred_test, y_test, score_dict, model_name, targets)
        # Save model
        mlflow.sklearn.log_model(opt_model_dict, mlflow_model_name)
    return(y_model_dict, opt_model_dict, score_dict)

