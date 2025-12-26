import mlflow

mlflow.set_experiment('Quickstart tutorial')

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the dataset into train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# define model hyperparameters
params = {
    'solver': 'lbfgs',
    'max_iter':2000,
    'multi_class':'auto',
    'random_state':1
}

# enable autologging for scikit-learn
mlflow.autolog()

# train model
print(f'Training model...')
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
print(f'Model training completed...')