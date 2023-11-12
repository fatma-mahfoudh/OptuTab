import optuna
import mlflow
from utils.utils_eval import log_results

def instantiate_xgboost(trial, random_state, task, objective):
    """Create XGBoost model to tune

    Args:
        trial (Trial): Trial Optuna object
        random_state (int): seed number
        task (string): regression or classification
        objective (string): learning objective, see xgboost documentation for list
    """
    params = {
            "objective": objective,
            "n_estimators": trial.suggest_int("n_estimators", 3, 30),
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 0.1),
            "random_state": random_state,
            "max_leaves": trial.suggest_int("max_leaves", 6, 20),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            # "early_stopping_rounds":50
        }
    if task == "regression":
        from xgboost import XGBRegressor
        return(XGBRegressor(**params))
    elif task == "classification":
        from xgboost import XGBClassifier
        return(XGBClassifier(**params))
    else:
        raise Exception(f"Task should be in (regression, classification), {task} passed") 

def instantiate_model(trial, model_name, random_state, task, objective=None):
    """Create model instance to tune

    Args:
        trial (Trial): Trial Optuna object
        model_name (string): model name
        random_state (int): seed number
        task (string): regression or classification
        objective (string): check xgb documentation for list 

    Raises:
        Exception: exception if model name not in [XGBoost]
    """
    model_name = model_name.lower()
    if model_name in ["xgboost"]:
        model = getattr(sys.modules[__name__], f"instantiate_{model_name}")(trial, random_state, task, objective)
    else:
        raise Exception(f"model name should be in [xgboost], {model_name} was passed")
    return(model)

def objective(trial, model_name, X, y, n_splits, random_state, task, objective=None, score_function):
    """Define objective function for hyperparameter optimization

    Args:
        trial (Trial): Trial Optuna Object
        model_name (string): model name
        X (DataFrame): Training Features
        y (DataFrame or Series): Training Targets
        n_splits (int): number of splits for cross validation
        random_state (int): random seed
        task (string): regression or classification
        objective (string): see xgb documentation for list
        score_function (function): function for scoring
    """
    model = instantiate_model(trial, model_name, random_state, task, objective)
    if model_name == "xgboost":
        from xgboost import cv, DMatrix
        from optuna.integration import XGBoostPruningCallback
        dtrain = DMatrix(X, label=y)
        pruning_callback = XGBoostPruningCallback(trial, f"test-{score_function.__name__}")
        cv_scores = cv(model.get_params(), dtrain, nfold=n_splits, stratified=True, feval=score_function, early_stopping_rounds=30, callbacks=[pruning_callback], seed=random_state)
        return(cv_scores["test-" + score_function.__name__ + "-mean"].values[-1])
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mse_scorer = make_scorer(mean_squared_error)
        scores = cross_val_score(model, X, y, scoring=mse_scorer, cv=kf)
        return(np.min([np.mean(scores), np.median(scores)]))

def callback(study, trial):
    """Callback function to save best model

    Args:
        study (Study): Optuna Study object
        trial (Trial): Optuna Trial object
    """
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def tune_model(run_name, model_name, direction, task, n_trials, X_train, y_train, X_test, y_test, n_splits, 
               random_state, mlflow_model_name):
    """Tune model and plot results

    Args:
        run_name (string): mlflow run name
        model_name (string): model name
        direction (string): maximize or minimize
        task (string): regression or classification
        n_trials (int): number of trials
        X_train (DataFrame): training features
        y_train (DataFrame): training targets
        X_test (DataFrame): testing features
        y_test (DataFrame): testing targets
        n_splits (int): number of splits for cross validation
        random_state (int): number to fix seed 
        objective (string): see xgb documentation for list
        score_function (function): function for scoring
        mlflow_model_name (string): model name to be used for mlflow logging
    """
    with mlflow.start_run(run_name=run_name):
        y_pred_train = pd.DataFrame([])
        y_pred_test = pd.DataFrame([])
        opt_model_dict = {}
        for target in targets:
            print(f'Started tuning {model_name} for {target}...')
            study = optuna.create_study(direction=direction, study_name=task, 
                                        sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True))
            study.optimize(lambda trial: objective(trial, model_name, X_train, y_train[target], n_splits, random_state), 
                        n_trials=n_trials, callbacks=[callback])
            print(f'Tuning finished! \n')
            opt_model = study.user_attrs["best_booster"]
            opt_model.fit(X_train, y_train[target])
            y_pred_train_target = pd.DataFrame(opt_model.predict(X_train), columns=[target])
            y_pred_test_target = pd.DataFrame(opt_model.predict(X_test), columns=[target])
            opt_model_dict[target] = opt_model
            y_pred_train = pd.concat([y_pred_train, y_pred_train_target], axis=1)
            y_pred_test = pd.concat([y_pred_test, y_pred_test_target], axis=1)
        y_model_dict = log_results(y_pred_train, y_train, y_pred_test, y_test, model_name, targets)
        # Save model
        mlflow.sklearn.log_model(opt_model_dict, mlflow_model_name)
    return(y_model_dict, opt_model_dict)