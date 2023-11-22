import optuna
import mlflow
from helpers.helper_eval import log_results
import pandas as pd
import numpy as np
import sys
import os
import contextlib

def instantiate_xgboost(params, task, objective, random_state):
    """Creates XGBoost model

    Args:
        params (dictionary): dictionary of xgboost params
        task (string): regression or classification
        objective (string): see XGBoost documentation
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        XGBoost instance: XGBoost model instance specific to task
    """
    default_params = {
            "objective": objective,
            "random_state": random_state,
            # "grow_policy": "lossguide",
            # "tree_method": "hist",
            # "max_depth": 0,
            "verbosity": 0,
        }
    params.update(default_params)
    if task == "regression":
        from xgboost import XGBRegressor
        return(XGBRegressor(**params))
    elif task == "classification":
        from xgboost import XGBClassifier
        return(XGBClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")
    
def instantiate_lgbm(params, task, random_state, **kwargs):
    """Creates LGBM model

    Args:
        params (dictionary): dictionary of lgbm params
        task (string): regression or classification
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        LGBM instance: LGBM model instance specific to task
    """
    default_params = {
            "objective": task,
            "random_state": random_state,
            "verbosity": -1,
        }
    params.update(default_params)
    if task == "regression":
        from lightgbm import LGBMRegressor
        return(LGBMRegressor(**params))
    elif task == "classification":
        from lightgbm import LGBMClassifier
        return(LGBMClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")
    
def instantiate_catboost(params, task, random_state, **kwargs):
    """Creates CatBoost model

    Args:
        params (dictionary): dictionary of xgboost params
        task (string): regression or classification
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        CatBoost instance: CatBoost model instance specific to task
    """
    default_params = {
            "random_state": random_state,
            "allow_writing_files":False,
            "silent":True
        }
    params.update(default_params)
    if task == "regression":
        from catboost import CatBoostRegressor
        return(CatBoostRegressor(**params))
    elif task == "classification":
        from catboost import CatBoostClassifier
        return(CatBoostClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")
    
def instantiate_randomforest(params, task, random_state, **kwargs):
    """Creates Random Forest model

    Args:
        params (dictionary): dictionary of Random Forest params
        task (string): regression or classification
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        Random Forest instance: Random Forest model instance specific to task
    """
    default_params = {
            "random_state": random_state,
            "verbose" : 0
            }
    params.update(default_params)
    if task == "regression":
        from sklearn.ensemble import RandomForestRegressor
        return(RandomForestRegressor(**params))
    elif task == "classification":
        from sklearn.ensemble import RandomForestClassifier
        return(RandomForestClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")

def instantiate_extratrees(params, task, random_state, **kwargs):
    """Creates Extra Trees model

    Args:
        params (dictionary): dictionary of Extra Trees params
        task (string): regression or classification
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        Extra Trees instance: Extra Trees model instance specific to task
    """
    default_params = {
            "random_state": random_state,
            "verbose" : 0
            }
    params.update(default_params)
    if task == "regression":
        from sklearn.ensemble import ExtraTreesRegressor
        return(ExtraTreesRegressor(**params))
    elif task == "classification":
        from sklearn.ensemble import ExtraTreesClassifier
        return(ExtraTreesClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")
    
def instantiate_svm(params, task, **kwargs):
    """Creates SVM model

    Args:
        params (dictionary): dictionary of SVM params
        task (string): regression or classification
        random_state (int): seed number

    Raises:
        Exception: raised if task not in list

    Returns:
        Extra Trees instance: Extra Trees model instance specific to task
    """
    default_params = {
            "verbose" : 0
            }
    params.update(default_params)
    if task == "regression":
        from sklearn.svm import SVR #LinearSVR for large datasets
        return(SVR(**params))
    elif task == "classification":
        from sklearn.svm import SVC #LinearSVC for large datasets
        return(SVC(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")

def instantiate_knn(params, task, **kwargs):
    """Creates KNearestNeighbors model

    Args:
        params (dictionary): dictionary of KNearestNeighbors params
        task (string): regression or classification

    Raises:
        Exception: raised if task not in list

    Returns:
        KNearestNeighbors instance: Extra Trees model instance specific to task
    """
    if task == "regression":
        from sklearn.neighbors import KNeighborsRegressor
        return(KNeighborsRegressor(**params))
    elif task == "classification":
        from sklearn.neighbors import KNeighborsClassifier
        return(KNeighborsClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed")

def instantiate_elasticnet(params, random_state, **kwargs):
    """Creates ElasticNet model

    Args:
        params (dictionary): dictionary of elasticnet params
        random_state (int): seed number

    Returns:
        ElasticNet instance: ElasticNet model instance
    """
    from sklearn.linear_model import ElasticNet
    default_params = {
                        "random_state": random_state
                     }
    params.update(default_params)
    return(ElasticNet(**params))

def instantiate_trial_xgboost(trial, random_state, task, objective):
    """Creates trial XGBoost model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification
        objective (string): learning objective, see xgboost documentation for list

    Returns:
        XGBoost instance: XGBoost model instance with Trial parameters
    """
    #TODO take range of hyperparameters as input
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.6, log=True),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 1),
            "reg_alpha": trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 2, 20),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 6), 
            "early_stopping_rounds":10

        }
    # params["max_depth"] = trial.suggest_int("max_depth", 3, 6) if params["tree_method"] == "exact" else 0
    return(instantiate_xgboost(params, task, objective=objective, random_state=random_state))

def instantiate_trial_lgbm(trial, random_state, task, **kwargs):
    """Creates trial LGBM model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification

    Returns:
        LGBM instance: LGBM model instance with Trial parameters
    """
    params = {
        'random_state': random_state,
        'n_estimators': trial.suggest_int('n_estimators', 3, 30), #100
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 6),
        'num_leaves' : trial.suggest_int('num_leaves', 6, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
    }
    return(instantiate_lgbm(params, task, random_state=random_state)) 

def instantiate_trial_catboost(trial, random_state, task, **kwargs):
    """Creates trial CatBoost model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification

    Returns:
        CatBoost instance: CatBoost model instance with Trial parameters
    """
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.6, log=True),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "reg_lambda": trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            # "max_leaves": trial.suggest_int("max_leaves", 2, 20),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 6), 
            "early_stopping_rounds":10

        }
    return(instantiate_catboost(params, task, random_state=random_state))

def instantiate_trial_randomforest(trial, random_state, task, **kwargs):
    """Creates trial Random Forest model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification

    Returns:
        Random Forest instance: Random Forest model instance with Trial parameters
    """
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 3, 15),
            "max_depth": trial.suggest_int("max_depth", 3, 6), 
            "min_samples_split": trial.suggest_categorical("min_samples_split", [0.6, 0.7, 0.8, 0.9, 1.0]),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.1, 0.5),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 20),
        }
    return(instantiate_randomforest(params, task, random_state=random_state))

def instantiate_trial_extratrees(trial, random_state, task, **kwargs):
    """Creates trial Extra Trees model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification

    Returns:
        Extra Trees instance: Extra Trees model instance with Trial parameters
    """
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 3, 15),
            "max_depth": trial.suggest_int("max_depth", 3, 6), 
            "min_samples_split": trial.suggest_categorical("min_samples_split", [0.6, 0.7, 0.8, 0.9, 1.0]),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.1, 0.5),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 20),
        }
    return(instantiate_extratrees(params, task, random_state=random_state))

def instantiate_trial_svm(trial, task, **kwargs):
    """Creates trial SVM model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification

    Returns:
        SVM instance: SVM model instance with Trial parameters
    """
    params = {
            "C": trial.suggest_float("C", 1e-2, 10),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    return(instantiate_svm(params, task))

def instantiate_trial_knn(trial, task, **kwargs):
    """Creates trial KNearestNeighbors model to tune

    Args:
        trial (Trial): Trial Optuna object
        task (string): regression or classification

    Returns:
        KNearestNeighbors instance: KNearestNeighbors model instance with Trial parameters
    """
    params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 7),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]), 
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        }
    return(instantiate_knn(params, task))

def instantiate_trial_elasticnet(trial, random_state, **kwargs):
    """Creates trial ElasticNet model to tune

    Args:
        trial (Trial): Trial Optuna Object
        random_state (int): seed number

    Returns:
        ElasticNet instance: ElasticNet model instance with Trial parameters
    """
    params = {
        "alpha": trial.suggest_float("alpha", 1e-2, 10),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 1),   
            }
    default_params = {
        "random_state": random_state
            }
    params.update(default_params)
    return(instantiate_elasticnet(params, random_state))

def instantiate_model(trial, model_name, random_state, task, xgb_objective=None):
    """Creates model instance to tune

    Args:
        trial (Trial): Trial Optuna object
        model_name (string): model name
        random_state (int): seed number
        task (string): regression or classification
        xgb_objective (string): check xgb documentation for list 

    Raises:
        Exception: exception if model name not in ["xgboost", "lgbm", "catboost", "elasticnet", "randomforest", "extratrees", "knn", "svm"]

    Returns:
        model instance: model instance with Trial parameters
    """
    model_name = model_name.lower()
    supported_models = ["xgboost", "lgbm", "catboost", "elasticnet", "randomforest", "extratrees", "knn", "svm"]
    if model_name in supported_models:
        model = getattr(sys.modules[__name__], f"instantiate_trial_{model_name}")(trial=trial, random_state=random_state, task=task, 
                                                                                  objective=xgb_objective)
    else:
        raise Exception(f"model name should be in {supported_models}, {model_name} was passed")
    return(model)

def objective(trial, model_name, X, y, n_splits, random_state, task, base_score_function, 
              cv_score_function, xgb_objective=None):
    """Defines objective function for hyperparameter optimization

    Args:
        trial (Trial): Trial Optuna Object
        model_name (string): model name
        X (DataFrame): Training Features
        y (DataFrame or Series): Training Targets
        n_splits (int): number of splits for cross validation
        random_state (int): random seed
        task (string): regression or classification
        base_score_function (function): base function for scoring
        cv_score_function (function): cross validation function for scoring LGBM, XGBoost
        xgb_objective (string): see xgb documentation for list

    Returns:
        float: objective function value to optimize
    """
    model = instantiate_model(trial, model_name, random_state, task, xgb_objective)
    if model_name == "xgboost":
        from xgboost import cv, DMatrix
        from optuna.integration import XGBoostPruningCallback
        dtrain = DMatrix(X, label=y)
        pruning_callback = XGBoostPruningCallback(trial, f"test-{cv_score_function.__name__}")
        cv_scores = cv(model.get_params(), dtrain, nfold=n_splits, 
                       stratified=True if task == "classification" else False, 
                       custom_metric=cv_score_function, early_stopping_rounds=10, 
                       callbacks=[pruning_callback], shuffle=True,
                       seed=random_state)
        return(cv_scores[f"test-{cv_score_function.__name__}-mean"].values[-1])#, cv_scores["test-" + score_function.__name__ + "-std"].values[-1])
    elif model_name == "lgbm":
        from lightgbm import cv, Dataset
        from optuna.integration import LightGBMPruningCallback
        dtrain = Dataset(data=X, label=y)
        pruning_callback = LightGBMPruningCallback(trial, f"valid {cv_score_function.__name__}")
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            cv_scores = cv(params=model.get_params(), train_set=dtrain, 
                        nfold=n_splits,
                        stratified=True if task == "classification" else False,
                        feval=cv_score_function, seed=random_state,
                        callbacks=[pruning_callback], shuffle=True
                        )
        return(cv_scores[f"valid {cv_score_function.__name__}-mean"][-1])
    elif model_name == "catboost":
        from catboost import cv, Pool
        from optuna.integration import CatBoostPruningCallback
        cv_data = Pool(data=X, label=y)
        pruning_callback = CatBoostPruningCallback(trial, f"test-{base_score_function}")
        cv_scores = cv(pool=cv_data, params=model.get_params(), nfold=n_splits, 
                    seed=random_state, shuffle=True, 
                    stratified=True if task == "classification" else False, 
                    verbose=False, early_stopping_rounds=10, plot=False)
        return(cv_scores[f"test-{base_score_function}-mean"].values[-1])
    else:
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.metrics import make_scorer
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mse_scorer = make_scorer(base_score_function)
        scores = cross_val_score(model, X, y, scoring=mse_scorer, cv=kf)
        return(np.min([np.mean(scores), np.median(scores)]))

def tune_model(run_name, model_name, direction, timeout, targets, X_train, y_train, X_test, y_test, n_splits, 
               random_state, base_score_function, mlflow_model_name, task=None, cv_score_function=None,
               inverse_transform=None, warm_start_dict=None, xgb_objective=None):
    """Tunes model and plot results

    Args:
        run_name (string): mlflow run name
        model_name (string): model name
        direction (string): maximize or minimize
        timeout (int): tuning budget time allocated for each target in seconds 
        targets (list): list of target names
        X_train (DataFrame): training features
        y_train (DataFrame): training targets
        X_test (DataFrame): testing features
        y_test (DataFrame): testing targets
        n_splits (int): number of splits for cross validation
        random_state (int): number to fix seed 
        base_score_function (function): base function for cv scoring
        mlflow_model_name (string): model name to be used for mlflow logging
        task (string): regression or classification
        cv_score_function (function): cross validation function for scoring LGBM or XGBoost
        inverse_transform (function): function to inverse transform targets
        warm_start_dict (dictionary): dictionary of list of params dictionaries to start optimization with for each target {"target1":[{"param1":val1, "param2":val2},...]}
        xgb_objective (string): see xgb documentation for list

    Returns:
        dictionary: score dictionary 
        dictionary: ground truth and prediction dictionary for train and test splits
        dictionary: best model dictionary (keys are target names)
    """
    score_dict = {key:{} for key in targets} # + ['all']
    with mlflow.start_run(run_name=run_name):
        y_pred_train = pd.DataFrame([])
        y_pred_test = pd.DataFrame([])
        opt_model_dict = {}
        for target in targets:
            y_train_target = y_train[target]
            print(f'Started tuning {model_name} for {target}...')
            study = optuna.create_study(direction=direction, study_name=task, 
                                        sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True))
            if warm_start_dict and target in warm_start_dict.keys():
                warm_start_list = warm_start_dict[target]
                for i in range(len(warm_start_list)):
                    study.enqueue_trial(warm_start_list[i])
            study.optimize(lambda trial: objective(trial, model_name, X_train, y_train_target, n_splits, 
                                                   random_state, task, base_score_function, cv_score_function, xgb_objective), 
                        timeout=timeout)
            print(f'Tuning finished! \n')
            best_params = study.best_params #max(study.best_trials, key=lambda t: t.values[1]).params #multi objective optimization
            opt_model = getattr(sys.modules[__name__], f"instantiate_{model_name}")(best_params, task=task, 
                                                                                    objective=xgb_objective, 
                                                                                    random_state=random_state)
            opt_model.fit(X_train, y_train_target)
            y_pred_train_target = pd.DataFrame(opt_model.predict(X_train), columns=[target])
            y_pred_test_target = pd.DataFrame(opt_model.predict(X_test), columns=[target])
            y_train_target = pd.DataFrame(y_train_target, columns=[target])
            y_test_target = pd.DataFrame(y_test[target], columns=[target])
            if inverse_transform:
                y_pred_train_target = inverse_transform(y_pred_train_target)
                y_pred_test_target = inverse_transform(y_pred_test_target)
                y_train_target = inverse_transform(y_train_target)
                y_test_target = inverse_transform(y_test_target)
            score_dict = log_results(score_dict, y_pred_train_target, y_train_target, y_pred_test_target, y_test_target, target, model_name, task)
            opt_model_dict[target] = opt_model
            y_pred_train = pd.concat([y_pred_train, y_pred_train_target], axis=1)
            y_pred_test = pd.concat([y_pred_test, y_pred_test_target], axis=1)
        y_model_dict = {
        'train':{
            'pred':y_pred_train,
            'true':inverse_transform(y_train) if inverse_transform else y_train
        },
        'test':{
            'pred':y_pred_test,
            'true':inverse_transform(y_test) if inverse_transform else y_test
        }
        }
        # Log model name
        mlflow.log_params({"model_name": model_name})
        # Save model
        mlflow.sklearn.log_model(opt_model_dict, mlflow_model_name)
    return(score_dict, y_model_dict, opt_model_dict)