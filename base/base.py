
import pandas as pd 
import numpy as np
import requests
import io

def load_trainset(mode = 'local'):
    if mode == 'local':
        train_path = 'data/train.npz'
        data = np.load(train_path)
    elif mode == 'remote':
        train_path = 'http://82.174.108.246:9999/files/train'
        response = requests.get(train_path)
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content)) 

    X_train = pd.DataFrame(data['X_train'], columns = data['feature_names'])
    y_train = data['y_train']
    return X_train, y_train
        
def load_testset(mode = 'local'):
    if mode == 'local':
        train_path = 'data/test.npz'
        data = np.load(train_path)
    elif mode == 'remote':
        train_path = 'http://82.174.108.246:9999/files/test'
        response = requests.get(train_path)
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content)) 

    X_test = pd.DataFrame(data['X_test'], columns = data['feature_names'])
    y_test = data['y_test']
    return X_test, y_test

def dataset_split(file_path  = '../data/DATASET_THESIS_2022.csv.gz'):
    import tqdm
    from sklearn.model_selection import train_test_split

    df_data = pd.concat([chunk for chunk in tqdm(pd.read_csv(file_path, chunksize=1000, compression = 'gzip'), desc='Loading data')])

    feat_names = [col.replace(' ', '') for col in df_data.columns]
    df_data.columns = feat_names
    feat_names.remove('TARGET')
    y = df_data.TARGET
    X = df_data.drop(columns=['TARGET'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state= 42)

    np.savez_compressed(
        file = '../data/train',
        X_train = X_train,
        y_train = y_train,
        feature_names = feat_names
    )

    np.savez_compressed(
        file = '../data/test',
        X_test = X_test,
        y_test = y_test,
        feature_names = feat_names
    )

def summary_table(df):

  print(f'Dataset Shape: {df.shape}')
  
  summary = pd.DataFrame(df.dtypes, columns = ['dtypes'])
  summary = summary.reset_index()
  summary['Name'] = summary['index']
  summary = summary[['Name', 'dtypes']]
  summary['Missing_Ratio'] = df.isnull().sum().values / len(df)
  summary['Uniques_Num'] = df.nunique().values
  summary['Uniques_Ratio'] = summary['Uniques_Num']/ len(df)
  summary['Mean'] = df.mean().values

  return summary

