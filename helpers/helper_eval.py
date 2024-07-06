import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, cross_validate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def rmse(y_true, y_pred):
    """Root Mean Squared Error function

    Args:
        y_true (DataFrame): ground truth target data
        y_pred (DataFrame): prediction data 

    Returns:
        float: RMSE value
    """
    return(mean_squared_error(y_true, y_pred, squared=False))

def f1_micro(y_true, y_pred):
    """F1 score with micro average

    Args:
        y_true (DataFrame): ground truth target data
        y_pred (DataFrame): prediction data

    Returns:
        float: F1 score value
    """
    return(f1_score(y_true, y_pred, average="micro"))

def f1_macro(y_true, y_pred):
    """F1 score with macro average

    Args:
        y_true (DataFrame): ground truth target data
        y_pred (DataFrame): prediction data

    Returns:
        float: F1 score value
    """
    return(f1_score(y_true, y_pred, average="macro"))

def f1_weighted(y_true, y_pred):
    """F1 score with weighted average

    Args:
        y_true (DataFrame): ground truth target data
        y_pred (DataFrame): prediction data

    Returns:
        float: F1 score value
    """
    return(f1_score(y_true, y_pred, average="weighted"))

def add_score(score_dict, y_dict, model_name, target_name, task):
    """Update the score dict for each new model

    Args:
        score_dict (dictionary): dictionary of RMSE/MAE scores
        y_dict (dictionary): dictionary of predictions and groung truth
        model_name (string): name of ML model
        target name (string): target name
        task (string): regression or classification

    Returns:
        dictionary: score dictionary
    """
    if task == "regression":
        score_fc = {
            "RMSE": rmse,
            "MAE": mean_absolute_error,
            "R2": r2_score
        }
    elif task == "classification":
        score_fc = {
            "Accuracy": accuracy_score,
            "F1-micro": f1_micro,
            "F1-macro": f1_macro,
            "F1-weighted": f1_weighted
        }
    else:
        raise Exception(f"Task should be either regression or classification, {task} was passed")
    score_dict[target_name][model_name] = {key:{} for key in score_fc.keys()}
                                
    for key in score_fc.keys():
        for split in ["train", "test"]:
            fc = score_fc[key]
            score_dict[target_name][model_name][key][split] = fc(y_dict[split]['true'][target_name], y_dict[split]['pred'][target_name])
    
    return(score_dict)

def log_results(score_dict, y_pred_train, y_train, y_pred_test, y_test, target_name, model_name, task):
    """Log results in MLflow and display plots

    Args:
        score_dict (dict): score dictionary
        y_pred_train (DataFrame): predictions on training set
        y_train (DataFrame): ground truth of training set 
        y_pred_test (DataFrame): predictions on test set
        y_test (DataFrame): ground truth on test set
        target_name (string): target name
        model_name (string): model name
        task (string): regression or classification

    Returns:
        dictionary: score dictionary
    """
    y_bench_dict = {
        'train':{
            'pred':y_pred_train,
            'true':y_train
        },
        'test':{
            'pred':y_pred_test,
            'true':y_test
        }
    }
    # Add scores
    score_dict = add_score(score_dict, y_bench_dict, model_name, target_name, task)
    for key in score_dict[target_name][model_name].keys():
        for split in ["train", "test"]:
            mlflow.log_metric(f"{target_name}-{key}-{split}", score_dict[target_name][model_name][key][split])
    return(score_dict)

def cv_results(X, y, groups, models, n_splits, task):
    """Returns cross validation results of each model passed in models 

    Args:
        X (DataFrame): training data
        y (DataFrame): target data
        groups (Series): groups in data
        models (list): list of sets [(model_name, model_instance)]
        n_splits (int): number of folds
        task (string): regression or classification

    Raises:
        Exception: raised if task value is not expected

    Returns:
        dictionary: cross validation result dictionary
    """
    if task == "regression":
        score_fc = {
            "RMSE": [rmse, "neg_root_mean_squared_error"],
            "MAE": [mean_absolute_error, "neg_mean_absolute_error"],
            "R2": [r2_score, "r2"],
        }
    elif task == "classification":
        score_fc = {
            "Accuracy": [accuracy_score, "accuracy"],
            "F1-micro": [f1_micro, "f1_macro"],
            "F1-macro": [f1_macro, "f1_micro"],
            "F1-weighted": [f1_weighted, "f1_weighted"],
        }
    else:
        raise Exception(f"Task should be either regression or classification, {task} was passed")
    scoring = [value[1] for value in score_fc.values()]
    results_details = {key: {name: None for name, _ in models} for key in score_fc.keys()}

    for name, model in models:
        if groups is None:
            kfold = KFold(n_splits=n_splits)
            cv_results = cross_val_predict(model, X, y, cv=kfold)
            cv_results_details = cross_validate(model, X, y, cv=kfold,
                                                scoring=scoring)
        else:
            kfold = GroupKFold(n_splits=n_splits)
            cv_results = cross_val_predict(model, X, y, cv=kfold, groups=groups)
            cv_results_details = cross_validate(model, X, y, cv=kfold, groups=groups,
                                                scoring=scoring)
        msg = f"{name}: "
        for key in results_details.keys():
            results_details[key][name] = cv_results_details[f"test_{score_fc[key][1]}"] * ((-1) ** ("neg" in score_fc[key][1]))
            msg += f"{key} {score_fc[key][0](y, cv_results):.4f}, "
        print(msg)
    return results_details

def display_cv_results(results_details):
    """Displays results of cross validation

    Args:
        results_details (dictionary): cross validation dictionary results
    """
    df_results = pd.DataFrame([])
    for i, key in enumerate(results_details.keys()):
        df_score = pd.DataFrame.from_dict(results_details[key]).melt()
        df_score.rename(columns={"variable":"model", 
                                 "value":key}, inplace=True)
        if i == 0:
            df_results = pd.concat([df_results, df_score], axis=1)
        else:
            df_results = pd.concat([df_results, df_score[[key]]], axis=1)

    agg_mean = df_results.groupby("model").mean().reset_index()
    agg_mean.columns = ["model"] + [f"mean_{key}" for key in results_details.keys()]
    df_results = df_results.merge(agg_mean, left_on="model", right_on="model")

    fig, ax = plt.subplots(len(results_details), 1, figsize=(18, 5 * len(results_details)))
    for i, key in enumerate(results_details.keys()):
        df_results.sort_values(by=f"mean_{key}", inplace=True, ascending=False)
        sns.boxplot(x="model", y=key, data=df_results, ax=ax[i])
        ax[i].tick_params(axis='y', labelsize=15)
        ax[i].tick_params(axis='x', labelsize=15, rotation=90)
        ax[i].set_ylabel(key, fontsize=15)
        ax[i].set_xlabel('')

    plt.tight_layout()
    plt.show()
