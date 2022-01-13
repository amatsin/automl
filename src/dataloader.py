import pickle

import pandas as pd
from autofeat import AutoFeatRegressor
from sklearn.preprocessing import StandardScaler


def load_data(scale=True, n_train_rows=None, autofeat_transform=False):
    print("Reading training data...")
    original_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')[:n_train_rows]
    print("Original train shape: ",  original_train.shape)

    freq_enc_train_features = frequency_encoding(original_train)

    if autofeat_transform:
        train = pd.concat([add_autofeat_features(original_train), freq_enc_train_features], axis=1)
    else:
        train = pd.concat([original_train(original_train), freq_enc_train_features], axis=1)

    if scale:
        train = scale_data(train)
    print('target data type is', train['target'].dtype)
    print('shape returned by load_data function', train.shape)
    return train


def add_autofeat_features(data):
    """
    Inputs:
        - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
    Returns:
        - new_df: new pandas dataframe with all the original features (except categorical features transformed
                  into multiple 0/1 columns) and the most promising engineered features.
    """
    with open('autofeat_regressor.pickle', mode='rb') as fp:
        autofeat_transformer: AutoFeatRegressor = pickle.load(fp)
    with_original_and_new_features = autofeat_transformer.transform(
        data.drop(['ID_code', 'target'], axis=1).values.astype(float))
    result = pd.concat([with_original_and_new_features, data.ID_code, data.target], axis=1)
    print('shape returned by add_autofeat_features function', result.shape)
    return result


def scale_data(train):
    print('Scaling data...')
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    train[X_columns] = X_train

    return train


def frequency_encoding(train):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv
    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    train_df = pd.DataFrame()
    for col in idx:
        train_df[col + '_freq'] = train[col].map(train.groupby(col).size())
        train_df = train_df.copy()

    print('shape returned by frequency_encoding function', train_df.shape)
    return train_df
