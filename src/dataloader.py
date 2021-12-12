import pandas as pd
from sklearn.preprocessing import StandardScaler

def balance_data(train):
    # TODO: is there better way to sample?
    train_true = train.loc[train['target'] == 1]
    print("Train target true length: ", len(train_true))
    train_false = train.loc[train['target'] != 1].sample(frac=1)[:len(train_true)]
    print("Train target false length: ", len(train_false))
    train = pd.concat([train_true, train_false], ignore_index=True).sample(frac=1)
    print("Train balanced length: ", len(train))
    return train

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

def load_data():
    print("Reading training data")
    train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
    print("Train length: ", len(train))
    train = balance_data(train)
    test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    print("Test length: ", len(test))
    train, test = scale_data(train, test)
    return train, test
