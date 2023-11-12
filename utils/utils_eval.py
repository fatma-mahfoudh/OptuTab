import mlflow

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
    return(y_bench_dict)