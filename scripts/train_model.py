import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

def train_model():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())

    dump(model, 'models/model.joblib')

if __name__ == "__main__":
    train_model()
