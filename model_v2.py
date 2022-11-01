import sys


ENV_COLAB = 'google.colab' in sys.modules
FILE_NAME = 'DATASET_THESIS_2022.csv'

if ENV_COLAB:

  DIR = '/content/drive/My Drive/Thesis Project/'
  from google.colab import output, drive

  output.enable_custom_widget_manager()
  drive.mount('/content/drive/')

else:

  DIR = './'

from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score   
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler,
    RobustScaler,
    )

# for feature engineering
from feature_engine import imputation as mdi
import lightgbm as lgb

from hyperopt import fmin, tpe, Trials
from hyperopt import STATUS_OK


def objective_function(params):

    params_dict = {
        'class_weight': 'balanced',  
        'learning_rate': params['learning_rate'],
        'max_depth': int(params['max_depth']),
        'n_estimators': int(params['n_estimators']),
        'num_leaves': int(params['num_leaves']),
        'boosting_type': params['boosting_type'],
        'colsample_bytree': params['colsample_bytree'],
        'reg_lambda': params['reg_lambda']
      }

    clf = lgb.LGBMClassifier(**params_dict)

    score = cross_val_score(
        clf, X_train, y_train, 
        cv = StratifiedKFold(n_splits= 5, shuffle =True), scoring ='neg_log_loss', n_jobs= 3
        ).mean()
    
    return {'loss': -score, 'status': STATUS_OK}



if __name__ == '__main__':

    df_data = pd.concat([chunk for chunk in tqdm(pd.read_csv(DIR + FILE_NAME, chunksize=1000), desc='Loading data')])
    y = df_data.TARGET
    X = df_data.drop(columns=['TARGET'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state= 42)

    pipe = Pipeline([
                    ('imputer', mdi.MeanMedianImputer(imputation_method='median')),
                    ('scaler', StandardScaler()),
                ])
    X_train = pipe.fit_transform(X_train)
    X_test =  pipe.transform(X_test)


    
    # Domain space
    space = {
        'learning_rate' : hp.loguniform('learning_rat', np.log(0.1), np.log(1)),
        'max_depth': hp.quniform('max_depth', 5, 15, 1),
        'n_estimators': hp.quniform('n_estimators', 5, 100, 1),
        'num_leaves': hp.quniform('num_leaves', 5, 25, 1),
        'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.3, 1.0),
        'reg_lambda':hp.uniform('reg_lambda', 0.0, 1.0)
    }


    # Optimization algorithm
    tpe_algorithm = tpe.suggest


    # Result history 
    trials = Trials()
    num_eval = 100

    print('begain to tune')
    best_params = fmin(
        objective_function, space, tpe.suggest, max_evals = num_eval, 
        trials = trials, rstate = np.random.RandomState(0)
        )
    print('finish tuning')
        
    pickle.dump(best_params, 'best_params.pkl')
    pickle.dump(trials, 'results.pkl')

