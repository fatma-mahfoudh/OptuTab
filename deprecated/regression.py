import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('dark_background')

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def cv_results(X, y, groups, models):
    results_rmse = []
    results_r2 = []

    results_details_rmse = {}
    results_details_r2 = {}

    for name, model in models:
        kfold = GroupKFold(n_splits=10)
        cv_results = cross_val_predict(model, X, y, groups, cv=kfold)
        cv_results_details = cross_validate(model, X, y, cv=kfold,
                                            scoring=['neg_mean_squared_error', 'r2'])

        results_rmse.append(np.sqrt(mean_squared_error(y, cv_results)))
        results_r2.append(r2_score(y, cv_results))

        results_details_rmse[name] = cv_results_details['test_neg_mean_squared_error'].apply(lambda x: np.sqrt(x))
        results_details_r2[name] = cv_results_details['test_r2']

        msg = "{}: RMSE {}, R2 {}".format(name,
                                          round(np.sqrt(mean_squared_error(y,
                                                                           cv_results)),
                                                4),
                                          round(r2_score(y, cv_results),
                                                4))
        print(msg)
    return results_details_rmse, results_details_r2


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

    # fig, ax = plt.subplots(2, 2, figsize=(32, 18))
    #
    # results_df.sort_values(by='mean_acc', inplace=True)
    # sns.boxplot(x='model', y='acc', data=results_df[results_df['mean_acc'] >= 0.8], ax=ax[0, 0])
    # ax[0, 0].tick_params(axis='y', labelsize=15)
    # ax[0, 0].tick_params(axis='x', labelsize=15, rotation=90)
    # ax[0, 0].set_ylabel('Accuracy', fontsize=15)
    # ax[0, 0].set_xlabel('')
    #
    # results_df.sort_values(by='mean_f1_micro', inplace=True)
    # sns.boxplot(x='model', y='f1_micro', data=results_df[results_df['mean_f1_micro'] >= 0.8], ax=ax[0, 1])
    # ax[0, 1].tick_params(axis='y', labelsize=15)
    # ax[0, 1].tick_params(axis='x', labelsize=15, rotation=90)
    # ax[0, 1].set_ylabel('F1 micro', fontsize=15)
    # ax[0, 1].set_xlabel('')
    #
    # plt.tight_layout()
    # plt.show()


def display_results(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print('RMSE - train: {} %'.format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
    print('RMSE - val: {} %'.format(np.sqrt(mean_squared_error(y_val, y_val_pred))))
    print('RMSE - test: {} % \n'.format(np.sqrt(mean_squared_error(y_test, y_test_pred))))

    print('R2 - train: {}'.format(r2_score(y_train, y_train_pred)))
    print('R2 - val: {}'.format(r2_score(y_val, y_val_pred)))
    print('R2 - test: {} \n'.format(r2_score(y_test, y_test_pred)))

    # ft_imp = pd.DataFrame([])
    # ft_imp['ft'] = X_train.columns
    # ft_imp['imp'] = model.feature_importances_
    # ft_imp.sort_values(by='imp', inplace=True)
    # sns.barplot(x='ft', y='imp', data=ft_imp)
    # plt.title('Feature Importance - {} model'.format(model_name))
    # plt.show()


def fine_tune_catb(X, y, groups, gpu):
    '''
#     returns best Light Gradient Boosting Model obtained by tuning parameters with Randomized search CV
#     Inputs:
#         X_train: dataframe, features in the training set
#         X_val: dataframe, features in the validation set
#         y_train: dataframe/series, target in the training set
#         y_val: dataframe/series, target in the validation set
#     '''
    if gpu:
        catb_model = CatBoostRegressor(loss_function='RMSE',
                                       n_estimators=100,
                                       task_type="GPU",
                                       #bootstrap_type='Poisson',
                                       #grow_policy='Lossguide',
                                       random_state=2021)
    else:
        catb_model = CatBoostRegressor(loss_function='RMSE',
                                       n_estimators=100,
                                       #bootstrap_type='Poisson',
                                       #grow_policy='Lossguide',
                                       random_state=2021)
    params = {'learning_rate': [1e-3, 1e-2, 1e-1],
              #'n_estimators': [100, 200, 300],
              'max_depth': [10, 15, 20, 25, 30],
              #'max_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              #'subsample': sp_uniform(loc=0.2, scale=0.8),
              #'colsample_bylevel': sp_uniform(loc=0.4, scale=0.6),
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    grid = RandomizedSearchCV(
        estimator=catb_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2021,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X,
             y,
             groups=groups,
             # eval_metric='rmse',
             #early_stopping_rounds=10,
             verbose=False)
    print('best parameters: {}'.format(grid.best_params_))
    if gpu:
        catb_model = CatBoostRegressor(loss_function='RMSE',
                                       n_estimators=100,
                                       task_type="GPU",
                                       #bootstrap_type='Bernoulli',
                                       #grow_policy='Lossguide',
                                       random_state=2021,
                                       **grid.best_params_)
    else:
        catb_model = CatBoostRegressor(loss_function='RMSE',
                                       n_estimators=100,
                                       #bootstrap_type='Bernoulli',
                                       #grow_policy='Lossguide',
                                       random_state=2021,
                                       **grid.best_params_)
    # catb_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
    catb_model.fit(X, y)
    # joblib.dump(catb_model, os.path.join('..','models','catboost','run_objective_met','catb_{}.pkl'.format(id_)))

    return catb_model


def fine_tune_xgb(X, y, groups, gpu):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    if gpu:
        xgb_model = XGBRegressor(n_jobs=-1,
                                 tree_method='gpu_hist',
                                 gpu_id=0,
                                 predictor='gpu_predictor',
                                 n_estimators=100,
                                 # max_depth=30,
                                 random_state=2021)
    else:
        xgb_model = XGBRegressor(n_jobs=-1,
                                 # max_depth=30,
                                 n_estimators=100,
                                 random_state=2021)
    params = {'learning_rate': [1e-3, 1e-2, 1e-1],
              'max_depth': [10, 15, 20, 25, 30],
              #'n_estimators': [100, 200, 300],
              'max_leaves': sp_randint(6, 50),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
              }

    grid = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2021,
        verbose=False)
    #     grid = GridSearchCV(
    #                         estimator=xgb_model,
    #                         param_grid=params,
    # #                         n_iter=100,
    #                         scoring='f1_macro',
    #                         cv=3,
    #                         refit=True,
    #                         n_jobs=-1,
    # #                         random_state=2020,
    #                         verbose=False)
    print('\n tuning started...')

    grid.fit(X, y,
             groups=groups,
             # val_metric='mlogloss',
             #early_stopping_rounds=10,
             verbose=False)
    print('best parameters: {}'.format(grid.best_params_))

    xgb_model = XGBRegressor(n_jobs=-1,
                             n_estimators=100,
                             random_state=2021,
                             **grid.best_params_)
    #     xgb_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
    xgb_model.fit(X, y)

    return xgb_model


def fine_tune_lgbm(X, y, groups, gpu):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    if gpu:
        lgbm_model = LGBMRegressor(
            device='gpu',
            n_jobs=-1,
            random_state=2021,
            n_estimators=100,
            # max_depth=30, #10,
        )
    else:
        lgbm_model = LGBMRegressor(
            n_jobs=-1,
            random_state=2021,
            n_estimators=100,
            # max_depth=30, #10,
        )
    params = {
        'max_depth': [10, 15, 20, 25, 30],
        #'n_estimators': [100, 200, 300],
        'learning_rate': [1e-3, 1e-2, 1e-1],
        'num_leaves': sp_randint(6, 80),  # sp_randint(6, 50),
        'min_child_samples': sp_randint(100, 500),
        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
        'subsample': sp_uniform(loc=0.2, scale=0.8),
        'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
        'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
    }

    grid = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2021,
        # verbose=False
    )
    print('\n tuning started...')

    grid.fit(X,
             y,
             groups=groups,
             # eval_metric='multi_logloss',
             #early_stopping_rounds=10,
             # verbose=False
             )
    print('best parameters: {}'.format(grid.best_params_))
    if gpu:
        lgbm_model = LGBMRegressor(n_jobs=-1,
                                   device='gpu',
                                   random_state=2021,
                                   n_estimators=100,
                                   # max_depth=10,
                                   **grid.best_params_)
    else:
        lgbm_model = LGBMRegressor(n_jobs=-1,
                                   random_state=2021,
                                   n_estimators=100,
                                   # max_depth=10,
                                   **grid.best_params_)
    #     lgbm_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
    lgbm_model.fit(X, y)

    return lgbm_model


def fine_tune_rf(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    rf_model = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=2021)
    params = {
              #'n_estimators': [int(x) for x in np.linspace(start=50, stop=100, num=3)],  # 100,500,10 #100,1000,10
              'max_features': ['auto', 'sqrt'],
              'max_depth': [int(x) for x in np.linspace(5, 10, num=6)],  # 15 #20
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]
              }

    grid = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2020,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    rf_model = RandomForestRegressor(
        n_jobs=-1,
        n_estimators=100,
        random_state=2021,
        **grid.best_params_)
    rf_model.fit(X, y)

    return rf_model


def fine_tune_extratrees(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    et_model = ExtraTreesRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=2021)
    params = {
              #'n_estimators': [int(x) for x in np.linspace(start=50, stop=100, num=3)],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [int(x) for x in np.linspace(5, 15, num=10)],  # 20 #10,110,11
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]
              }

    grid = RandomizedSearchCV(
        estimator=et_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2021,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    et_model = ExtraTreesRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=2021,
        **grid.best_params_)
    et_model.fit(X, y)

    return et_model


def fine_tune_decisiontree(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    tree_model = DecisionTreeRegressor(random_state=2021)
    params = {
        # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [int(x) for x in np.linspace(5, 10, num=6)],  # 5, 15, 10 #10, 110, 11
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
    }
    grid = GridSearchCV(
        estimator=tree_model,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=False)

    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    tree_model = DecisionTreeRegressor(random_state=2021, **grid.best_params_)
    tree_model.fit(X, y)

    return tree_model


def fine_tune_gb(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    gb_model = GradientBoostingRegressor(
        random_state=2021)
    params = {
        #              'loss':['deviance','exponential'],
        'learning_rate': [1e-3, 1e-2, 1e-1],
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=100, num=3)],
        'subsample': sp_uniform(loc=0.2, scale=0.8),
        #               'criterion':['friedman_mse', 'mse'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        #               'max_features': ['auto', 'sqrt'],
    }

    grid = RandomizedSearchCV(
        estimator=gb_model,
        param_distributions=params,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=2021,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    gb_model = GradientBoostingRegressor(
        random_state=2021,
        **grid.best_params_)
    gb_model.fit(X, y)

    return gb_model


def fine_tune_knn(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    knn_model = KNeighborsRegressor(n_jobs=-1)
    params = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              }

    grid = GridSearchCV(
        estimator=knn_model,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    knn_model = KNeighborsRegressor(**grid.best_params_)
    knn_model.fit(X, y)

    return knn_model


def fine_tune_svr(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    svr_model = SVR()
    params = {
        'C': [1e-2, 0.1, 1, 10],  # [1e-2, 5 * 1e-2, 0.1, 1, 5, 10, 20],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }

    grid = GridSearchCV(
        estimator=svr_model,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    svr_model = SVR(**grid.best_params_)
    svr_model.fit(X, y)

    return svr_model


def fine_tune_lr(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    lr_model = LinearRegression(n_jobs=-1)
    params = {'normalize': [True, False]}
    grid = GridSearchCV(
        estimator=lr_model,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    lr_model = LinearRegression(**grid.best_params_)
    lr_model.fit(X, y)

    return lr_model


def fine_tune_ridge(X, y, groups):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    ridge_model = Ridge(random_state=2021)
    params = {
        'alpha': [0.1, 1, 5, 10, 20]}
    grid = GridSearchCV(
        estimator=ridge_model,
        param_grid=params,
        scoring='neg_mean_squared_error',
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=False)
    print('\n tuning started...')

    grid.fit(X, y, groups=groups)
    print('best parameters: {}'.format(grid.best_params_))

    ridge_model = Ridge(random_state=2021, **grid.best_params_)
    ridge_model.fit(X, y)

    return ridge_model


def cv_results_tuned(X, y, models, groups, n_splits, gpu):
    results_rmse = []
    results_r2 = []

    results_details_rmse = {}
    results_details_r2 = {}

    for name, model in models:
        if groups is None:
            kfold = KFold(n_splits=n_splits)
        else:
            kfold = GroupKFold(n_splits=n_splits)
        ##tuning
        if name == 'Decision Tree':
            model = fine_tune_decisiontree(X, y, groups)
        elif name == 'Extra Trees':
            model = fine_tune_extratrees(X, y, groups)
        elif name == 'Random Forest':
            model = fine_tune_rf(X, y, groups)
        elif name == 'KNN':
            model = fine_tune_knn(X, y, groups)
        elif name in ['SVM']:
            model = fine_tune_svr(X, y, groups)
        elif name in ['Ridge']:
            model = fine_tune_ridge(X, y, groups)
        elif name == 'Linear Regression':
            model = fine_tune_lr(X, y, groups)
        elif name == 'XGBoost':
            model = fine_tune_xgb(X, y, groups, gpu)
        elif name == 'LGBM':
            model = fine_tune_lgbm(X, y, groups, gpu)
        elif name == 'CatBoost':
            model = fine_tune_catb(X, y, groups, gpu)
        else:
            print('model name not in list')
        if groups is None:
            cv_results = cross_val_predict(model, X, y, cv=kfold)
            cv_results_details = cross_validate(model, X, y, cv=kfold,
                                                scoring=['neg_mean_squared_error', 'r2'])
        else:
            cv_results = cross_val_predict(model, X, y, cv=kfold, groups=groups)
            cv_results_details = cross_validate(model, X, y, cv=kfold, groups=groups,
                                                scoring=['neg_mean_squared_error', 'r2'])

        results_rmse.append(np.sqrt(mean_squared_error(y, cv_results)))
        results_r2.append(r2_score(y, cv_results))

        results_details_rmse[name] = np.sqrt(-cv_results_details['test_neg_mean_squared_error'])
        results_details_r2[name] = cv_results_details['test_r2']

        msg = "{}: RMSE {}, R2 {}".format(name,
                                          round(np.sqrt(mean_squared_error(y,
                                                                           cv_results)),
                                                4),
                                          round(r2_score(y, cv_results),
                                                4))
        print(msg)
    return results_details_rmse, results_details_r2
