# OptuTab
You will find in this repo a list of functions to fine tune state-of-the-art machine learning models using <b>Optuna</b>. Metrics are logged using <b>MLflow</b>. The supported models are: <b>LGBM, XGBoost, CatBoost, Random Forest, ExtraTrees, KNN, SVM, ElasticNet</b>.  
The user is encouraged to change the range of hyperparameters accordingly (e.g. number of estimators, max depth...)  

An example is given below:
```python
from tune_models import tune_model

mlflow_run_name =  "XGBoost - warmstart" 
model_name = "xgboost"
direction = "minimize"
task = "regression"
# time allocated for each target in seconds
timeout = 10*60 
mlflow_model_name = "xgboost"
n_splits = 3
random_state = 42
xgb_objective = "reg:squarederror"
# base score function in cross validation
base_score_function = mean_squared_error
# LGBM and XGBoost cross validation score function
def cv_score_function(y_true, y_pred):
    """Returns score

    Args:
        y_true (Series): ground truth data
        y_pred (DMatrix): DMatrix XGBoost object
    
    """
    return(base_score_function.__name__, base_score_function(y_true, y_pred.get_label()))
cv_score_function.__name__ = base_score_function.__name__
# warm start dictionary parameters for Optuna, Optional
warm_start_dict = {
    "target1":[
        {'n_estimators': 14, 'max_leaves': 6, 'min_child_weight': 3.7027037553361097, 'learning_rate': 0.40137638002469916, 'subsample': 0.8, 'colsample_bylevel': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.019611914193186634, 'reg_lambda': 0.059359693678740125}
    ]
}
targets = ["target1", "target2", "target3"]
# calling main function
score_dict, y_model_dict, opt_model_dict = tune_model(
            run_name=mlflow_run_name, 
            model_name=model_name,
            direction=direction,
            task=task,
            timeout=timeout,
            targets=targets,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test,
            n_splits=n_splits,
            random_state=random_state, 
            xgb_objective=xgb_objective,
            base_score_function=base_score_function,
            cv_score_function=cv_score_function,
            mlflow_model_name=mlflow_model_name,
            warm_start_dict=warm_start_dict
        )
```