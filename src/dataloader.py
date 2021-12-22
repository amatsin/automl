import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm


def scale_data(train, test):
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns
    X_test = test.drop(['ID_code'], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train[X_columns] = X_train
    test[X_columns] = X_test

    return train, test


def remove_synthetic(test):
    # Code taken from: https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
    print("Removing synthetic data from test")
    df_test = test.drop(['ID_code'], axis=1)

    df_test = df_test.values

    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    print(f'real_samples_indexes {len(real_samples_indexes)}')
    print(f'synthetic_samples_indexes {len(synthetic_samples_indexes)}')

    return test.iloc[~test.index.isin(synthetic_samples_indexes)]


def frequency_encoding(train, test):
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv

    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    traintest = pd.concat([train, test])
    traintest = traintest.reset_index(drop=True)

    for col in idx:
        traintest[col + '_freq'] = traintest[col].map(traintest.groupby(col).size())

    train_df = traintest[:len(train)]
    test_df = traintest[len(train):]
    test_df = test_df.drop(['target'], axis=1)

    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df


def load_data(scale=True):
    print("Reading training data")
    train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
    print("Train length: ", len(train))

    test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    train, test = frequency_encoding(train, test)

    test = remove_synthetic(test)
    print("Test length: ", len(test))
    if scale:
        train, test = scale_data(train, test)

    return train, test
