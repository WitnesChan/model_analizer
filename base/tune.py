import numpy as np
import lightgbm as lgb
import catboost  as cb
import xgboost as xgb 

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature_engine import imputation as mdi

import optuna
from sklearn.metrics import roc_auc_score
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback

from base import load_trainset
import util


def tune_lgb(trial_name = 'lgb_trials', n_trials = 100, device_type = 'gpu'):
    
    def objective(trial):
        ### define the hyper-parameter space
        param_grid = {
            "metric": "auc",
            "n_estimators": trial.suggest_int("n_estimators", 250, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 1e-5, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 1e-5, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
            'device_type': device_type
        }

        data, target = load_trainset(mode='local')
        ### preprocess features
        ### add later

        ### define the 5-fold cross-validation set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1112324)
        cv_scores = np.empty(5)
    
        for idx, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            model = lgb.LGBMClassifier(objective="binary", **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="binary_logloss",
                callbacks=[LightGBMPruningCallback(trial, "auc"), lgb.early_stopping(100)],  # Add a pruning callback
            )
            preds = model.predict_proba(X_test)[:,1]
            cv_scores[idx] = roc_auc_score(y_test, preds)

        ### average the 5 out-of-sample auc 
        return np.mean(cv_scores)
    

    # default sampler in Optuna Tree-structured Parzen Estimater (TPE)
    study = optuna.create_study(
        storage=util.sqlite_path, 
        study_name= trial_name, load_if_exists = True,
        pruner= optuna.pruners.MedianPruner(n_warmup_steps=10),
        direction="maximize"
    )
    
    study.optimize(objective, n_trials= n_trials)

    log_file = open('data/tune.log', mode = 'a')
    fmt_log_str = f"LGB: Best value: {study.best_value} (params: {study.best_params})"
    print(fmt_log_str)
    log_file.write(fmt_log_str)

def tune_xgb(trial_name = 'xgb_trials',  n_trials=100, device_type = 'cpu'):

    def objective(trial):
  
        ### define the hyper-parameter space
        param_grid = {
            "silent": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-3, 0.3, log = True),
            "alpha": trial.suggest_float("alpha", 1e-3, 0.8, log = True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "early_stopping_rounds": 30,
            "n_jobs": -1
        }

        if device_type == 'gpu':
            param_grid['gpu_id'] = 0 
            param_grid['tree_method'] = 'gpu_hist'

        if param_grid["booster"] == "gbtree" or param_grid["booster"] == "dart":
            param_grid["max_depth"] = trial.suggest_int("max_depth", 2, 4)
            param_grid["eta"] = trial.suggest_float("eta", 0.1, 0.2, log = True)
            param_grid["gamma"] = trial.suggest_float("gamma", 1e-2, 0.6, log = True)
            param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            param_grid["sampling_method"] = trial.suggest_categorical("sampling_method", ['uniform', 'gradient_based'])
            param_grid["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0, log =False)
            param_grid["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.6, 1.0, log =False)
            param_grid["colsample_bynode"] = trial.suggest_float("colsample_bynode", 0.6, 1.0, log =False)
        
        if param_grid["booster"] == "dart":
            param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-3, 0.6, log= True)
            param_grid["skip_drop"] = trial.suggest_float("skip_drop", 1e-3, 0.2, log= True)

        data, target = load_trainset(mode='local')
        ### preprocess features
        ### add later

        ### define the 5-fold cross-validation set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1112324)
        cv_scores = np.empty(5)
    
        for idx, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]
            
            model = xgb.XGBClassifier(**param_grid)

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
                # callbacks=[XGBoostPruningCallback(trial, "auc")],  # Add a pruning callback
            )
            preds = model.predict_proba(X_test)[:,1]
            cv_scores[idx] = roc_auc_score(y_test, preds)

        ### average the 5 out-of-sample auc 
        return np.mean(cv_scores)

    # default sampler in Optuna Tree-structured Parzen Estimater (TPE)
    study = optuna.create_study(
        storage=util.sqlite_path, 
        study_name= trial_name, load_if_exists = True,
        pruner= optuna.pruners.MedianPruner(n_warmup_steps=10),
        direction="maximize"
    )
    
    study.optimize(objective, n_trials= n_trials)

    log_file = open('../data/tune.log', mode = 'a')
    fmt_log_str = f"XGB: Best value: {study.best_value} (params: {study.best_params})"
    print(fmt_log_str)
    log_file.write(fmt_log_str)

def tune_cb(trial_name = 'cb_trials', n_trials=100):
    
    def objective(trial):

        ### define the hyper-parameter space

        param_grid = {
            'max_depth': trial.suggest_int('max_depth', 3, 16),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
            'n_estimators': trial.suggest_int('n_estimators', 20, 200),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log = True),
            'subsample': trial.suggest_float('subsample', 0.1, 0.8),
            'random_seed': 42,
            'task_type': 'GPU',
            'devices': '0',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'bootstrap_type': 'Poisson'
        }

        data, target = load_trainset(mode='local')
        ### preprocess features
        ### add later
        
        ### define the 5-fold cross-validation set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1112324)
        cv_scores = np.empty(5)
        
        for idx, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]
            
            model = cb.CatBoostClassifier(**param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose = False,
                early_stopping_rounds=30,
                # callbacks=[XGBoostPruningCallback(trial, "auc")],  # Add a pruning callback
            )
            preds = model.predict_proba(X_test)[:,1]
            cv_scores[idx] = roc_auc_score(y_test, preds)

        ### average the 5 out-of-sample auc 
        return np.mean(cv_scores)
    
    # default sampler in Optuna Tree-structured Parzen Estimater (TPE)
    study = optuna.create_study(
        storage=util.sqlite_path, 
        study_name= trial_name, load_if_exists = True,
        pruner= optuna.pruners.MedianPruner(n_warmup_steps=10),
        direction="maximize"
    )
    
    study.optimize(objective, n_trials= n_trials)

    log_file = open('data/tune.log', mode = 'a')
    fmt_log_str = f"CB: Best value: {study.best_value} (params: {study.best_params})"
    print(fmt_log_str)
    log_file.write(fmt_log_str)

def tune_rf(trial_name = 'rf_trials', n_trials = 100):
    
    def objective(trial):
        
        ### define the hyper-parameter space
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }

        data, target = load_trainset(mode='local')
        ### preprocess features
        ### add later

        ### define the 5-fold cross-validation set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1112324)
        cv_scores = np.empty(5)
        
        for idx, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]
            
            model = RandomForestClassifier(random_state= 42, **param_grid)

            pipe = Pipeline([
                ('imputer', mdi.MeanMedianImputer(imputation_method='median')),
            ])
            X_train = pipe.fit_transform(X_train)
            X_test =  pipe.transform(X_test)

            model.fit(X_train,y_train)
            preds = model.predict_proba(X_test)[:,1]
            cv_scores[idx] = roc_auc_score(y_test, preds)

        ### average the 5 out-of-sample auc 
        return np.mean(cv_scores)

    # default sampler in Optuna Tree-structured Parzen Estimater (TPE)
    study = optuna.create_study(
        storage=util.sqlite_path, 
        study_name= trial_name, load_if_exists = True,
        pruner= optuna.pruners.MedianPruner(n_warmup_steps=10),
        direction="maximize"
    )
    
    study.optimize(objective, n_trials= n_trials)
    
    log_file = open('data/tune.log', mode = 'a')
    fmt_log_str = f"RF: Best value: {study.best_value} (params: {study.best_params})"
    print(fmt_log_str)
    log_file.write(fmt_log_str)

def tune_logit(trial_name = 'logit_trials'):
    ### TO-DO
    pass

def check_gpu_support():
    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lgb.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu'}
        lgb.train(params, train_set=train_data)
        return True
    except Exception as e:
        return False

if __name__ == '__main__':
    tune_xgb(trial_name= 'xgb_trials_v1', n_trials = 100)
    # tune_lgb(trial_name= 'lgb_trials_v3', n_trials = 100, device_type= 'gpu')
    # tune_rf(trial_name= 'rf_trials_v1', n_trials = 100)
    # tune_cb(trial_name= 'cb_trials_v1', n_trials = 100)
    # tune_lgb(trial_name= 'lgb_trials_gpu_v1', n_trials = 100, device_type= 'gpu')
    # tune_rf(trial_name= 'rf_trials_gpu_v1', n_trials = 100)
    # tune_cb(trial_name= 'cb_trials_gpu_v1', n_trials = 100)
    
