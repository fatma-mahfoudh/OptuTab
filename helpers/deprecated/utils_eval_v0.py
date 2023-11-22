import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, cross_validate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def rmse(y_true, y_pred):
#     """Compute the Root Mean Squared Error 

#     Args:
#         y_pred (Dataframe or Series or Array): predictions
#         y_true (Dataframe or Series or Array): ground truth
#     """
#     return(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_score(y_dict, model_name, targets):
    """Update the score dict for each new model

    Args:
        score_dict (dictionary): dictionary of RMSE/MAE scores
        y_dict (dictionary): dictionary of predictions and groung truth
        model_name (string): name of ML model
    """
    score_dict = {key:{} for key in targets + ['all']}
    for key in score_dict:
        score_dict[key][model_name] = {
                                        'RMSE':{},
                                        'MAE':{},
                                        'R2':{}
                                    }
        if key != 'all':
            # RMSE
            score_dict[key][model_name]['RMSE']['train'] = mean_squared_error(y_dict['train']['true'][key], y_dict['train']['pred'][key], squared=False)
            score_dict[key][model_name]['RMSE']['test'] = mean_squared_error(y_dict['test']['true'][key], y_dict['test']['pred'][key], squared=False)
            # MAE
            score_dict[key][model_name]['MAE']['train'] = mean_absolute_error(y_dict['train']['true'][key], y_dict['train']['pred'][key])
            score_dict[key][model_name]['MAE']['test'] = mean_absolute_error(y_dict['test']['true'][key], y_dict['test']['pred'][key])
            # R2
            score_dict[key][model_name]['R2']['train'] = r2_score(y_dict['train']['true'][key], y_dict['train']['pred'][key])
        elif key == 'all':
            # RMSE
            score_dict[key][model_name]['RMSE']['train'] = mean_squared_error(y_dict['train']['true'], y_dict['train']['pred'], squared=False)
            score_dict[key][model_name]['RMSE']['test'] = mean_squared_error(y_dict['test']['true'], y_dict['test']['pred'], squared=False)
            # MAE
            score_dict[key][model_name]['MAE']['train'] = mean_absolute_error(y_dict['train']['true'], y_dict['train']['pred'])
            score_dict[key][model_name]['MAE']['test'] = mean_absolute_error(y_dict['test']['true'], y_dict['test']['pred'])
    return(score_dict)

def log_results(y_pred_train, y_train, y_pred_test, y_test, model_name, targets):
    """Log results in MLflow and display plots

    Args:
        y_pred_train (DataFrame): predictions on training set
        y_train (DataFrame): ground truth of training set 
        y_pred_test (DataFrame): predictions on test set
        y_test (DataFrame): ground truth on test set
        model_name (string): model name
        targets (list): list of targets
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
    score_dict = add_score(y_bench_dict, model_name, targets)
    for key in score_dict.keys():
        if key != 'all':
            mlflow.log_metric(f"{key}-R2-train", score_dict[key][model_name]['R2']['train'])
        mlflow.log_metric(f"{key}-RMSE-train", score_dict[key][model_name]['RMSE']['train'])
        mlflow.log_metric(f"{key}-RMSE-test", score_dict[key][model_name]['RMSE']['test'])
        mlflow.log_metric(f"{key}-MAE-train", score_dict[key][model_name]['MAE']['train'])
        mlflow.log_metric(f"{key}-MAE-test", score_dict[key][model_name]['MAE']['test'])
    mlflow.log_params({
        "model_name": model_name  # Log the model name
    })
    return(score_dict, y_bench_dict)

def add_score_bis(score_dict, y_dict, model_name, target_name):
    """Update the score dict for each new model

    Args:
        score_dict (dictionary): dictionary of RMSE/MAE scores
        y_dict (dictionary): dictionary of predictions and groung truth
        model_name (string): name of ML model
    """
    score_dict[target_name][model_name] = {
                                        'RMSE':{},
                                        'MAE':{},
                                        'R2':{}
                                    }
    # RMSE
    score_dict[target_name][model_name]['RMSE']['train'] = mean_squared_error(y_dict['train']['true'][target_name], y_dict['train']['pred'][target_name], squared=False)
    score_dict[target_name][model_name]['RMSE']['test'] = mean_squared_error(y_dict['test']['true'][target_name], y_dict['test']['pred'][target_name], squared=False)
    # MAE
    score_dict[target_name][model_name]['MAE']['train'] = mean_absolute_error(y_dict['train']['true'][target_name], y_dict['train']['pred'][target_name])
    score_dict[target_name][model_name]['MAE']['test'] = mean_absolute_error(y_dict['test']['true'][target_name], y_dict['test']['pred'][target_name])
    # R2
    score_dict[target_name][model_name]['R2']['train'] = r2_score(y_dict['train']['true'][target_name], y_dict['train']['pred'][target_name])
    return(score_dict)

def log_results_bis(score_dict, y_pred_train, y_train, y_pred_test, y_test, target_name, model_name):
    """Log results in MLflow and display plots

    Args:
        y_pred_train (DataFrame): predictions on training set
        y_train (DataFrame): ground truth of training set 
        y_pred_test (DataFrame): predictions on test set
        y_test (DataFrame): ground truth on test set
        target_name (string): target name
        model_name (string): model name
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
    score_dict = add_score_bis(score_dict, y_bench_dict, model_name, target_name)
    mlflow.log_metric(f"{target_name}-R2-train", score_dict[target_name][model_name]['R2']['train'])
    mlflow.log_metric(f"{target_name}-RMSE-train", score_dict[target_name][model_name]['RMSE']['train'])
    mlflow.log_metric(f"{target_name}-RMSE-test", score_dict[target_name][model_name]['RMSE']['test'])
    mlflow.log_metric(f"{target_name}-MAE-train", score_dict[target_name][model_name]['MAE']['train'])
    mlflow.log_metric(f"{target_name}-MAE-test", score_dict[target_name][model_name]['MAE']['test'])
    return(score_dict)

def cv_results(X, y, groups, models, n_splits):
    results_rmse = []
    results_mae = []
    results_r2 = []

    results_details_rmse = {}
    results_details_mae = {}
    results_details_r2 = {}

    for name, model in models:
        if groups is None:
            kfold = KFold(n_splits=n_splits)
        else:
            kfold = GroupKFold(n_splits=n_splits)
        if groups is None:
            cv_results = cross_val_predict(model, X, y, cv=kfold)
            cv_results_details = cross_validate(model, X, y, cv=kfold,
                                                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
        else:
            cv_results = cross_val_predict(model, X, y, cv=kfold, groups=groups)
            cv_results_details = cross_validate(model, X, y, cv=kfold, groups=groups,
                                                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])

        results_rmse.append(mean_squared_error(y, cv_results, squared=False))
        results_mae.append(mean_absolute_error(y, cv_results))
        results_r2.append(r2_score(y, cv_results))

        results_details_rmse[name] = np.sqrt(-cv_results_details['test_neg_mean_squared_error'])
        results_details_mae[name] = np.sqrt(-cv_results_details['test_neg_mean_absolute_error'])
        results_details_r2[name] = cv_results_details['test_r2']

        msg = f"{name}: RMSE {mean_squared_error(y, cv_results, squared=False):.4f}, \
                        MAE {mean_absolute_error(y, cv_results):.4f}, \
                        R2 {r2_score(y, cv_results):.4f}"
        print(msg)
    return results_details_rmse, results_details_mae, results_details_r2

def display_cv_results(rmse, r2):
    results_df = pd.DataFrame.from_dict(rmse)
    results_df = results_df.melt(value_vars=results_df.columns)
    results_df.columns = ['model', 'RMSE']
    results_df.sort_values(by='model', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    col_names = ['R2']
    for i, results in enumerate([r2]):
        sub_results_df = pd.DataFrame.from_dict(results)
        sub_results_df = sub_results_df.melt(value_vars=sub_results_df.columns)
        sub_results_df.columns = ['model', col_names[i]]
        sub_results_df.sort_values(by='model', inplace=True)
        sub_results_df.reset_index(drop=True, inplace=True)
        results_df[col_names[i]] = sub_results_df[col_names[i]]
    agg_mean = results_df.groupby('model').mean().reset_index()
    agg_mean.columns = ['model', 'mean_RMSE', 'mean_R2']
    results_df = results_df.merge(agg_mean, left_on='model', right_on='model')

    fig, ax = plt.subplots(2, 1, figsize=(32, 18))

    results_df.sort_values(by='mean_RMSE', inplace=True, ascending=False)
    sns.boxplot(x='model', y='RMSE', data=results_df, ax=ax[0])
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=15, rotation=90)
    ax[0].set_ylabel('RMSE', fontsize=15)
    ax[0].set_xlabel('')

    results_df.sort_values(by='mean_R2', inplace=True)
    sns.boxplot(x='model', y='R2', data=results_df, ax=ax[1])
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=15, rotation=90)
    ax[1].set_ylabel('R2', fontsize=15)
    ax[1].set_xlabel('')

    plt.tight_layout()
    plt.show()
