import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

import xgboost as xgb
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import pickle

from tqdm import tqdm

# define the hyperparameter space

param_grid = [
    Integer(200, 2500, name='n_estimators'),
    Integer(1, 10, name='max_depth'),
    Real(0.01, 0.99, name='learning_rate'),
    Categorical(['gbtree', 'dart'], name='booster'),
    Real(0.01, 10, name='gamma'),
    Real(0.50, 0.90, name='subsample'),
    Real(0.50, 0.90, name='colsample_bytree'),
    Real(0.50, 0.90, name='colsample_bylevel'),
    Real(0.50, 0.90, name='colsample_bynode'),
    Integer(1, 50, name='reg_lambda'),
]


# set up the gradient boosting classifier

gbm = xgb.XGBClassifier(random_state=53)

# define the objective function
# We design a function to maximize the accuracy, of a GBM,
# with cross-validation

@use_named_args(param_grid)
def objective(**params):
    
    # model with new parameters
    gbm.set_params(**params)

    # optimization function (hyperparam response function)
    value = np.mean(
        cross_val_score(
            gbm, 
            X_train,
            y_train,
            cv=3,
            n_jobs=-4,
            scoring='accuracy')
    )

    # negate because we need to minimize
    return -value

# gp_minimize performs by default GP Optimization 
# sequential model-based optimization. Here we use Gaussian process-based Optimization.


file_name = 'DATASET_THESIS_2022.csv'


# if __name__ == '__main__':


df_data = pd.concat([chunk for chunk in tqdm(pd.read_csv(file_name, chunksize=1000), desc='Loading data')]).sample(10000)

print(df_data.shape)

y = df_data.TARGET

X = df_data.drop(columns=['TARGET'])

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state= 42)


gp_ = gp_minimize(
    objective, # the objective function to minimize
    param_grid, # the hyperparameter space
    n_initial_points=10, # the number of points to evaluate f(x) to start of
    acq_func='EI', # the acquisition function
    n_calls=40, # the number of subsequent evaluations of f(x)
    random_state=0
)


print("""Best parameters:
=========================
- n_estimators = %d
- max_depth = %d
- learning_rate = %.6f
- booster = %s
- gamma = %.6f
= subsample = %
- colsample_bytree = %.6f
- colsample_bylevel = %.6f
- colsample_bynode' = %.6f
""" % (gp_.x[0],
    gp_.x[1],
    gp_.x[2],
    gp_.x[3],
    gp_.x[4],
    gp_.x[5],
    gp_.x[6],
    gp_.x[7],
    gp_.x[8],
    ))

pickle.dump(gp_, open('./gp_tune.res', 'wb'))
print('done')
