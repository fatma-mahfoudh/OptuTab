import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import mlflow
mlflow.autolog(disable=True)
import os


def rmse(y_true, y_pred):
    """Compute the Root Mean Squared Error 

    Args:
        y_pred (Dataframe or Series or Array): predictions
        y_true (Dataframe or Series or Array): ground truth
    """
    return(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_score(original_score_dict, y_dict, model_name):
    """Update the score dict for each new model

    Args:
        score_dict (dictionary): dictionary of RMSE/MAE scores
        y_dict (dictionary): dictionary of predictions and groung truth
        model_name (string): name of ML model
    """
    score_dict = original_score_dict.copy()
    for key in score_dict:
        score_dict[key][model_name] = {
            'RMSE':{},
            'MAE':{},
            'R2':{}
        }
        if key != 'all':
            # RMSE
            score_dict[key][model_name]['RMSE']['train'] = rmse(y_dict['train']['true'][key], y_dict['train']['pred'][key])
            score_dict[key][model_name]['RMSE']['test'] = rmse(y_dict['test']['true'][key], y_dict['test']['pred'][key])
            # MAE
            score_dict[key][model_name]['MAE']['train'] = mean_absolute_error(y_dict['train']['true'][key], y_dict['train']['pred'][key])
            score_dict[key][model_name]['MAE']['test'] = mean_absolute_error(y_dict['test']['true'][key], y_dict['test']['pred'][key])
            # R2
            score_dict[key][model_name]['R2']['train'] = r2_score(y_dict['train']['true'][key], y_dict['train']['pred'][key])
        elif key == 'all':
            # RMSE
            score_dict[key][model_name]['RMSE']['train'] = rmse(y_dict['train']['true'], y_dict['train']['pred'])
            score_dict[key][model_name]['RMSE']['test'] = rmse(y_dict['test']['true'], y_dict['test']['pred'])
            # MAE
            score_dict[key][model_name]['MAE']['train'] = mean_absolute_error(y_dict['train']['true'], y_dict['train']['pred'])
            score_dict[key][model_name]['MAE']['test'] = mean_absolute_error(y_dict['test']['true'], y_dict['test']['pred'])
    return(score_dict)

def plot_comparison(y_subdict, names, split, score_dict, model_name):
    """Plot predictions vs ground truth for a specific split

    Args:
        y_subdict (dictionary): dictionary of predictions and ground truth
        names (list): feature names
        split (string): data split text should be in ['train', 'test']
        score_dict (dictionary): dictionary of scores
        model_name (string): name of the model
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    _, ax = plt.subplots(4, 4, figsize=(12,12))
    for i, name in enumerate(names):
        color_index = i % len(colors)  # Cycle through colors if more names than colors
        current_color = colors[color_index]
        ax[int(i//4), int(i%4)].scatter(y_subdict['true'][name],
                                     y_subdict['pred'][name], marker='o', s=20, alpha=0.7, c=current_color)
        ax[int(i//4), int(i%4)].plot([np.min([y_subdict['true'][name].values, y_subdict['pred'][name].values])-0.1, 
                                      np.max([y_subdict['true'][name].values, y_subdict['pred'][name].values])+0.1],
                                      [np.min([y_subdict['true'][name].values, y_subdict['pred'][name].values])-0.1, 
                                      np.max([y_subdict['true'][name].values, y_subdict['pred'][name].values])+0.1],
                                      color='red', linestyle='--')
        rmse = score_dict[name][model_name]['RMSE'][split]
        mae = score_dict[name][model_name]['MAE'][split]
        if split == 'train':
            r2 = score_dict[name][model_name]['R2'][split]
            ax[int(i//4), int(i%4)].set_title(f'{name} - R2:{r2:.1f} - RMSE:{rmse:.1f} - MAE:{mae:.1f}', fontsize=10)
        elif split == 'test':
            ax[int(i//4), int(i%4)].set_title(f'{name} - RMSE:{rmse:.1f} - MAE:{mae:.1f}',fontsize=10)
        ax[int(i // 4), int(i % 4)].grid(True, linestyle='--', alpha=0.6)
    plt.suptitle(split.upper())
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{model_name}-{split}.png'))
    plt.show()

def plot_results(y_dict, names, score_dict, model_name):
    """Plot predictions vs ground truth for all splits

    Args:
        y_dict (dictionary): dictionary of predictions and ground truth
        names (list): feature names
        score_dict (dictionary): dictionary of scores
        model_name (string): name of the model
    """
    plot_comparison(y_dict['train'], names, 'train', score_dict, model_name)
    plot_comparison(y_dict['test'], names, 'test', score_dict, model_name)

def log_results(y_pred_train, y_train, y_pred_test, y_test, score_dict, model_name, targets):
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
    score_dict = add_score(score_dict, y_bench_dict, model_name)
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
    # Plot results
    plot_results(y_bench_dict, targets, score_dict, model_name)
    return(score_dict, y_bench_dict)