## Create environment

```bash
# Sklearn environment
conda create -n santander -c conda-forge scikit-learn xgboost lightgbm catboost imbalanced-learn pandas six hyperopt python=3.8
conda activate santander
```

## Install Kaggle API

https://github.com/Kaggle/kaggle-api

```bash
pip install kaggle
# Download API token from 'Kaggle->Account->Create API Token' to ~/.kaggle folder
chmod 600 ~/.kaggle/kaggle.json
# Test API
kaggle competitions list
```

## Download dataset

```bash
cd input
kaggle competitions download -c santander-customer-transaction-prediction
unzip santander-customer-transaction-prediction.zip -d santander-customer-transaction-prediction
rm santander-customer-transaction-prediction.zip
```

## Running model

```bash
cd src
python LinearRegression.py
```

## Submit model to Kaggle

```bash
kaggle competitions submit -c santander-customer-transaction-prediction -f submission_LogisticRegression__folds5__0.611460871943926.csv -m "LogisticRegression"
```

## Weights and Biases setup

```bash
pip install wandb
wandb login
```