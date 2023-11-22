import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
plt.style.use('ggplot')

from sklearn.manifold import TSNE
# label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

#integrated multiclass
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC #(setting multi_class=”crammer_singer”)
from sklearn.linear_model import LogisticRegression, RidgeClassifier #(setting multi_class=”multinomial” to logistic regression) #LogisticRegressionCV #.RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
#one vs one
from sklearn.svm import NuSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier #(setting multi_class = “one_vs_one”)
#one vs rest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier #(setting multi_class = “one_vs_rest”)
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier

# from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

def cv_results(X, y, models):
    results_acc = []
    results_f1_micro = []
    results_f1_macro = []
    results_f1_weighted = []
    
    results_details_acc = {}
    results_details_f1_micro = {}
    results_details_f1_macro = {}
    results_details_f1_weighted = {}
    
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, shuffle=True)#KFold(n_splits=10, shuffle=True)
        cv_results = cross_val_predict(model, X, y, cv=kfold)
        cv_results_details = cross_validate(model, X, y, cv=kfold, scoring=['accuracy','f1_macro','f1_micro','f1_weighted'])#)
#         cv_results_details = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')#)
        
        results_acc.append(accuracy_score(y, cv_results))
        results_f1_micro.append(f1_score(y, cv_results, average='micro'))
        results_f1_macro.append(f1_score(y, cv_results, average='macro'))
        results_f1_weighted.append(f1_score(y, cv_results, average='weighted'))
        
        results_details_acc[name] = cv_results_details['test_accuracy']
        results_details_f1_micro[name] = cv_results_details['test_f1_micro']
        results_details_f1_macro[name] = cv_results_details['test_f1_macro']
        results_details_f1_weighted[name] = cv_results_details['test_f1_weighted']
#         msg = "{}: {} %".format(name, round(accuracy_score(y, cv_results) * 100, 3))
        msg = "{}: acc {} %, f1 micro {} %, f1 macro {} %, f1 weighted {} %".format(name,
                                             round(accuracy_score(y, cv_results) * 100, 3),
                                             round(f1_score(y, cv_results, average='micro') * 100, 3),
                                             round(f1_score(y, cv_results, average='macro') * 100, 3),
                                             round(f1_score(y, cv_results, average='weighted') * 100, 3))
        print(msg)
#     return(results_acc, results_f1_micro, results_f1_macro, results_f1_weighted, results_details_acc, results_details_f1, results_details_f1_macro, results_details_f1_weighted)
    return(results_details_acc, results_details_f1_micro, results_details_f1_macro, results_details_f1_weighted)

def display_cv_results(acc, f1_micro, f1_macro, f1_weighted):
    results_df = pd.DataFrame.from_dict(acc)
    results_df = results_df.melt(value_vars=results_df.columns)
    results_df.columns = ['model','acc']
    results_df.sort_values(by='model', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    col_names = ['f1_micro', 'f1_macro', 'f1_weighted']
    for i, results in enumerate([f1_micro, f1_macro, f1_weighted]):
        sub_results_df = pd.DataFrame.from_dict(results)
        sub_results_df = sub_results_df.melt(value_vars=sub_results_df.columns)
        sub_results_df.columns = ['model', col_names[i]]
        sub_results_df.sort_values(by='model', inplace=True)
        sub_results_df.reset_index(drop=True, inplace=True)
        results_df[col_names[i]] = sub_results_df[col_names[i]]
    agg_mean = results_df.groupby('model').mean().reset_index()
    agg_mean.columns = ['model','mean_acc', 'mean_f1_micro', 'mean_f1_macro', 'mean_f1_weighted']
    results_df = results_df.merge(agg_mean, left_on='model', right_on='model')
    
    fig, ax = plt.subplots(2, 2, figsize=(32,18))

    results_df.sort_values(by='mean_acc', inplace=True)
    sns.boxplot(x='model', y='acc', data=results_df, ax=ax[0,0])
    ax[0,0].tick_params(axis='y', labelsize=15)
    ax[0,0].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[0,0].set_ylabel('Accuracy', fontsize=15)
    ax[0,0].set_xlabel('')

    results_df.sort_values(by='mean_f1_micro', inplace=True)
    sns.boxplot(x='model', y='f1_micro', data=results_df, ax=ax[0,1])
    ax[0,1].tick_params(axis='y', labelsize=15)
    ax[0,1].tick_params(axis='x', labelsize=15, rotation=90)
    ax[0,1].set_ylabel('F1 micro', fontsize=15)
    ax[0,1].set_xlabel('')

    results_df.sort_values(by='mean_f1_macro', inplace=True)
    sns.boxplot(x='model', y='f1_macro', data=results_df, ax=ax[1,0])
    ax[1,0].tick_params(axis='y', labelsize=15)
    ax[1,0].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[1,0].set_ylabel('F1 macro', fontsize=15)
    ax[1,0].set_xlabel('models', fontsize=15)

    results_df.sort_values(by='mean_f1_weighted', inplace=True)
    sns.boxplot(x='model', y='f1_weighted', data=results_df, ax=ax[1,1])
    ax[1,1].tick_params(axis='y', labelsize=15)
    ax[1,1].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[1,1].set_ylabel('F1 weighted', fontsize=15)
    ax[1,1].set_xlabel('models', fontsize=15)

    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 2, figsize=(32,18))

    results_df.sort_values(by='mean_acc', inplace=True)
    sns.boxplot(x='model', y='acc', data=results_df[results_df['mean_acc'] >= 0.8], ax=ax[0,0])
    ax[0,0].tick_params(axis='y', labelsize=15)
    ax[0,0].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[0,0].set_ylabel('Accuracy', fontsize=15)
    ax[0,0].set_xlabel('')

    results_df.sort_values(by='mean_f1_micro', inplace=True)
    sns.boxplot(x='model', y='f1_micro', data=results_df[results_df['mean_f1_micro'] >= 0.8], ax=ax[0,1])
    ax[0,1].tick_params(axis='y', labelsize=15)
    ax[0,1].tick_params(axis='x', labelsize=15, rotation=90)
    ax[0,1].set_ylabel('F1 micro', fontsize=15)
    ax[0,1].set_xlabel('')

    results_df.sort_values(by='mean_f1_macro', inplace=True)
    sns.boxplot(x='model', y='f1_macro', data=results_df[results_df['mean_f1_macro'] >= 0.8], ax=ax[1,0])
    ax[1,0].tick_params(axis='y', labelsize=15)
    ax[1,0].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[1,0].set_ylabel('F1 macro', fontsize=15)
    ax[1,0].set_xlabel('models', fontsize=15)

    results_df.sort_values(by='mean_f1_weighted', inplace=True)
    sns.boxplot(x='model', y='f1_weighted', data=results_df[results_df['mean_f1_weighted'] >= 0.8], ax=ax[1,1])
    ax[1,1].tick_params(axis='y', labelsize=15)
    ax[1,1].tick_params(axis='x', labelsize=15, rotation=90) 
    ax[1,1].set_ylabel('F1 weighted', fontsize=15)
    ax[1,1].set_xlabel('models', fontsize=15)
    plt.suptitle('Perf > 0.8', fontsize=50, y=1)

    plt.tight_layout()
    plt.show()
    
    
def display_results(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test, le): #le: label encoder
    y_train_pred = model.predict(X_train)#.values)
    y_val_pred = model.predict(X_val)#.values)
    y_test_pred = model.predict(X_test)#.values)
    
    print('F1 score macro - train: {}'.format(f1_score(y_train, y_train_pred, average='macro')))
    print('F1 score macro - val: {}'.format(f1_score(y_val, y_val_pred, average='macro')))
    print('F1 score macro - test: {} \n'.format(f1_score(y_test, y_test_pred, average='macro')))
    
    print('Accuracy - train: {} %'.format(accuracy_score(y_train, y_train_pred) * 100))
    print('Accuracy - val: {} %'.format(accuracy_score(y_val, y_val_pred) * 100))
    print('Accuracy - test: {} % \n'.format(accuracy_score(y_test, y_test_pred) * 100))
    
    print('Precision score macro - train: {}'.format(precision_score(y_train, y_train_pred, average='macro')))
    print('Precision score macro - val: {}'.format(precision_score(y_val, y_val_pred, average='macro')))
    print('Precision score macro - test: {} \n'.format(precision_score(y_test, y_test_pred, average='macro')))
    
    print('Recall score macro - train: {}'.format(recall_score(y_train, y_train_pred, average='macro')))
    print('Recall score macro - val: {}'.format(recall_score(y_val, y_val_pred, average='macro')))
    print('Recall score macro - test: {} \n'.format(recall_score(y_test, y_test_pred, average='macro')))
    
    fig, ax = plt.subplots(1, 3, figsize=(30,5))
    plot_confusion_matrix(model, X_train, y_train, display_labels=le.classes_, normalize='true', cmap=plt.cm.GnBu, ax=ax[0])
    plot_confusion_matrix(model, X_val, y_val, display_labels=le.classes_, normalize='true', cmap=plt.cm.GnBu, ax=ax[1])
    plot_confusion_matrix(model, X_test, y_test, display_labels=le.classes_, normalize='true', cmap=plt.cm.GnBu, ax=ax[2])
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
    ax[2].set_title('Test')
    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    plt.show()
    
    ft_imp = pd.DataFrame([])
    ft_imp['ft'] = X_train.columns
    ft_imp['imp'] = model.feature_importances_
    ft_imp.sort_values(by='imp', inplace=True)
    sns.barplot(x='ft', y='imp', data=ft_imp)
    plt.title('Feature Importance - {} model'.format(model_name))
    plt.show()

# def fine_tune_catb(id_, X_train, X_val, y_train, y_val):
#     '''
#     returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
#     Inputs:
#         X_train: dataframe, features in the training set
#         X_val: dataframe, features in the validation set
#         y_train: dataframe/series, target in the training set
#         y_val: dataframe/series, target in the validation set
#     '''
#     catb_model = CatBoostClassifier(loss_function='RMSE',
#                                    n_estimators=100,
# #                                 task_type="GPU",
#                                    bootstrap_type='Bernoulli',
#                                    grow_policy='Lossguide',
#                                 random_state=2020)
#     params = {'learning_rate':[1e-3, 1e-2, 1e-1],
#             'max_leaves': sp_randint(6, 50), 
#              'min_child_samples': sp_randint(100, 500), 
#              'subsample': sp_uniform(loc=0.2, scale=0.8), 
# #              'colsample_bylevel': sp_uniform(loc=0.4, scale=0.6),
#              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#     grid = RandomizedSearchCV(
#                         estimator=catb_model,
#                         param_distributions=params, 
#                         n_iter=100,
#                         scoring='neg_mean_squared_error',
#                         cv=3,
#                         refit=True,
#                         n_jobs=10,
#                         random_state=2020,
#                         verbose=False)
#     print('Tuning started...')
    
#     grid.fit(X_train,
#                y_train.values,
#                eval_set=[(X_val, y_val.values)],
# #                eval_metric='rmse',
#                early_stopping_rounds=10,
#                verbose=False)
#     print('best parameters: {}'.format(grid.best_params_))

#     catb_model = CatBoostRegressor(loss_function='RMSE',
#                                    n_estimators=100,
# #                                 task_type="GPU",
#                                    bootstrap_type='Bernoulli',
#                                    grow_policy='Lossguide',
#                                 random_state=2020,
#                               **grid.best_params_)
#     catb_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
#     joblib.dump(catb_model, os.path.join('..','models','catboost','run_objective_met','catb_{}.pkl'.format(id_)))
    
#     return(catb_model)

def fine_tune_xgb(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    xgb_model = XGBClassifier(objective='multi:softmax', num_class=6,
                                n_jobs=10,
                                random_state=2021)
    params = {'learning_rate':[1e-3, 1e-2, 1e-1],
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
                        scoring='f1_macro',
                        cv=3,
                        refit=True,
                        n_jobs=10,
                        random_state=2021,
                        verbose=False)
#     grid = GridSearchCV(
#                         estimator=xgb_model,
#                         param_grid=params, 
# #                         n_iter=100,
#                         scoring='f1_macro',
#                         cv=3,
#                         refit=True,
#                         n_jobs=10,
# #                         random_state=2020,
#                         verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,
               y_train,
               eval_set=[(X_train, y_train), (X_val, y_val)],
               eval_metric='mlogloss',
               early_stopping_rounds=10,
               verbose=False)
    print('best parameters: {}'.format(grid.best_params_))

    xgb_model = XGBClassifier(objective='multi:softmax', num_class=6,
                                n_jobs=10,
                                random_state=2021,
                              **grid.best_params_)
#     xgb_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
    xgb_model.fit(X_train, y_train)
    
    return(xgb_model)


def fine_tune_lgbm(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    lgbm_model = LGBMClassifier(objective='multiclass',
                                n_jobs=10,
                                random_state=2020,
                                max_depth=10,
                                metric='multi_logloss')
    params = {
            'learning_rate':[1e-3, 1e-2, 1e-1],
            'num_leaves': sp_randint(6, 80), #sp_randint(6, 50), 
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
                        scoring='f1_macro',
                        cv=3,
                        refit=True,
                        n_jobs=10,
                        random_state=2020,
                        verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,
               y_train,
               eval_set=[(X_train, y_train), (X_val, y_val)],
               eval_metric='multi_logloss',
               early_stopping_rounds=10,
               verbose=False)
    print('best parameters: {}'.format(grid.best_params_))

    lgbm_model = LGBMClassifier(objective='multiclass',
                                n_jobs=10,
                                random_state=2021,
                                metric='multi_logloss',
                                max_depth=10,
                              **grid.best_params_)
#     lgbm_model.fit(np.concatenate([X_train,X_val], axis=0), np.concatenate([y_train,y_val], axis=0))
    lgbm_model.fit(X_train, y_train)
    
    return(lgbm_model)


def fine_tune_rf(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    rf_model = RandomForestClassifier(
                                n_jobs=10,
                                random_state=2020)
    params = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)], #100,500,10 #100,1000,10
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 10, num = 6)], #15 #20
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]
             }

    grid = RandomizedSearchCV(
                        estimator=rf_model,
                        param_distributions=params, 
                        n_iter=100,
                        scoring='f1_macro',
                        cv=3,
                        refit=True,
                        n_jobs=10,
                        random_state=2020,
                        verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    rf_model = RandomForestClassifier(
                                n_jobs=10,
                                random_state=2021,
                              **grid.best_params_)
    rf_model.fit(X_train, y_train)
    
    return(rf_model)

def fine_tune_extratrees(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    et_model = ExtraTreesClassifier(
                                n_jobs=10,
                                random_state=2020)
    params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 15, num = 10)], #20 #10,110,11
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]
             }

    grid = RandomizedSearchCV(
                        estimator=et_model,
                        param_distributions=params, 
                        n_iter=100,
                        scoring='f1_macro',
                        cv=3,
                        refit=True,
                        n_jobs=10,
                        random_state=2020,
                        verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    et_model = ExtraTreesClassifier(
                                n_jobs=10,
                                random_state=2021,
                              **grid.best_params_)
    et_model.fit(X_train, y_train)
    
    return(et_model)

def fine_tune_decisiontree(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    tree_model = DecisionTreeClassifier(random_state=2021)
    params = {'criterion':['gini','entropy'],
              'splitter':['best','random'],
              'max_depth': [int(x) for x in np.linspace(5, 10, num = 6)], #5, 15, 10 #10, 110, 11
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt','log2'],
            }
    grid = GridSearchCV(
                    estimator=tree_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)

    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    tree_model = DecisionTreeClassifier(random_state=2021, **grid.best_params_)
    tree_model.fit(X_train, y_train)
    
    return(tree_model)

def fine_tune_gb(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    gb_model = GradientBoostingClassifier(
                                random_state=2021)
    params = {
#              'loss':['deviance','exponential'],
              'learning_rate':[1e-3, 1e-2, 1e-1],
               'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
              'subsample': sp_uniform(loc=0.2, scale=0.8), 
#               'criterion':['friedman_mse', 'mse'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
#               'max_features': ['auto', 'sqrt'],
             }

    grid = RandomizedSearchCV(
                        estimator=gb_model,
                        param_distributions=params, 
                        n_iter=100,
                        scoring='f1_macro',
                        cv=3,
                        refit=True,
                        n_jobs=10,
                        random_state=2020,
                        verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    gb_model = GradientBoostingClassifier(
                                random_state=2021,
                              **grid.best_params_)
    gb_model.fit(X_train, y_train)
    
    return(gb_model)

def fine_tune_knn(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    knn_model = KNeighborsClassifier(n_jobs=10)
    params = {'n_neighbors':[3,5,7,9,11,13,15],
              'weights':['uniform','distance'],
              'algorithm':['auto','ball_tree','kd_tree','brute'],
             }

    grid = GridSearchCV(
                    estimator=knn_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    knn_model = KNeighborsClassifier(**grid.best_params_)
    knn_model.fit(X_train, y_train)
    
    return(knn_model)

def fine_tune_svc(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    svc_model = SVC(random_state=2021)
    params = {
              'C':[1e-2, 0.1, 1, 10], #[1e-2, 5 * 1e-2, 0.1, 1, 5, 10, 20],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
             }

    grid = GridSearchCV(
                    estimator=svc_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    svc_model = SVC(random_state=2021, **grid.best_params_)
    svc_model.fit(X_train, y_train)
    
    return(svc_model)


def fine_tune_qda(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    qda_model = QuadraticDiscriminantAnalysis()
    params = {'reg_param':[0.1, 0, 1, 5, 10]}
    grid = GridSearchCV(
                    estimator=qda_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    qda_model = QuadraticDiscriminantAnalysis(**grid.best_params_)
    qda_model.fit(X_train, y_train)
    
    return(qda_model)

def fine_tune_lr(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    lr_model = LogisticRegression(n_jobs=10, multi_class='multinomial', solver='newton-cg', penalty='l2', max_iter=1000)
    params = {'penalty':['l1', 'l2', 'elasticnet'],
             'C':[1e-2, 5 * 1e-2, 0.1, 1, 5, 10, 20]}
    grid = GridSearchCV(
                    estimator=lr_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    lr_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', **grid.best_params_)
    lr_model.fit(X_train, y_train)
    
    return(lr_model)

def fine_tune_ridge(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    ridge_model = RidgeClassifier(random_state=2021)
    params = {
             'alpha':[0.1, 1, 5, 10, 20]}
    grid = GridSearchCV(
                    estimator=ridge_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    ridge_model = RidgeClassifier(random_state=2021,**grid.best_params_)
    ridge_model.fit(X_train, y_train)
    
    return(ridge_model)


def fine_tune_radiusneighbor(X_train, X_val, y_train, y_val):
    '''
    returns best Light Gradient Boosting Model obtained by tuneing parameters with Randomized search CV
    Inputs:
        X_train: dataframe, features in the training set
        X_val: dataframe, features in the validation set
        y_train: dataframe/series, target in the training set
        y_val: dataframe/series, target in the validation set
    '''
    rn_model = RadiusNeighborsClassifier(n_jobs=10)
    params = {'radius':[15, 20, 25, 30],
              'weights':['uniform','distance'],
              'algorithm':['auto','ball_tree','kd_tree','brute'],
             }

    grid = GridSearchCV(
                    estimator=rn_model,
                    param_grid=params, 
                    scoring='f1_macro',
                    cv=3,
                    refit=True,
                    n_jobs=10,
                    verbose=False)
    print('Tuning started...')
    
    grid.fit(X_train,y_train)
    print('best parameters: {}'.format(grid.best_params_))

    rn_model = RadiusNeighborsClassifier(**grid.best_params_)
    rn_model.fit(X_train, y_train)
    
    return(rn_model)


def cv_results_tuned(X_train, X_val, y_train, y_val, models):
    results_acc = []
    results_f1_micro = []
    results_f1_macro = []
    results_f1_weighted = []
    
    results_details_acc = {}
    results_details_f1_micro = {}
    results_details_f1_macro = {}
    results_details_f1_weighted = {}
    
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        ##tuning
        if name == 'Gaussian Naive Bayes':
            model = model
        elif name == 'Decision Tree':
            model = fine_tune_decisiontree(X_train, X_val, y_train, y_val)
        elif name == 'Extra Trees':
            model = fine_tune_extratrees(X_train, X_val, y_train, y_val)
        elif name == 'Random Forest':
            model = fine_tune_rf(X_train, X_val, y_train, y_val)
        elif name == 'K-Nearest Neighbors':
            model = fine_tune_knn(X_train, X_val, y_train, y_val)
        elif name == 'Nearest Centroid':
            model = model
        elif name == 'Radius Neighbors Classifier':
            model = fine_tune_radiusneighbor(X_train, X_val, y_train, y_val)
        elif name == 'Quadratic Discriminant Analysis':
            model = fine_tune_qda(X_train, X_val, y_train, y_val)
        elif name in ['Support Vector Classifier', 'Support Vector Machine']:
            model = fine_tune_svc(X_train, X_val, y_train, y_val)
        elif name in ['Ridge Classifier', 'Multinomial Logistic Regression']:
            model = model
        elif name == 'XGBoost':
            model = fine_tune_xgb(X_train, X_val, y_train, y_val)
        elif name == 'LGBM':
            model = fine_tune_lgbm(X_train, X_val, y_train, y_val)
        else:
            print('model name not in list')
        X = X_train #np.concatenate([X_train,X_val])
        y = y_train #np.concatenate([y_train,y_val])
        cv_results = cross_val_predict(model, X, y, cv=kfold)
        cv_results_details = cross_validate(model, X, y, cv=kfold, scoring=['accuracy','f1_macro','f1_micro','f1_weighted'])#)
        
        results_acc.append(accuracy_score(y, cv_results))
        results_f1_micro.append(f1_score(y, cv_results, average='micro'))
        results_f1_macro.append(f1_score(y, cv_results, average='macro'))
        results_f1_weighted.append(f1_score(y, cv_results, average='weighted'))
        
        results_details_acc[name] = cv_results_details['test_accuracy']
        results_details_f1_micro[name] = cv_results_details['test_f1_micro']
        results_details_f1_macro[name] = cv_results_details['test_f1_macro']
        results_details_f1_weighted[name] = cv_results_details['test_f1_weighted']
#         msg = "{}: {} %".format(name, round(accuracy_score(y, cv_results) * 100, 3))
        msg = "{}: acc {} %, f1 micro {} %, f1 macro {} %, f1 weighted {} % \n".format(name,
                                             round(accuracy_score(y, cv_results) * 100, 3),
                                             round(f1_score(y, cv_results, average='micro') * 100, 3),
                                             round(f1_score(y, cv_results, average='macro') * 100, 3),
                                             round(f1_score(y, cv_results, average='weighted') * 100, 3))
        print(msg)
    return(results_details_acc, results_details_f1_micro, results_details_f1_macro, results_details_f1_weighted)