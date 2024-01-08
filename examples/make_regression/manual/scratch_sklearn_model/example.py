from transfernet import datasets, utils, models, plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import os


def main():

    # Parameters
    save_dir = './outputs'

    # Source training parameters
    n_epochs = 1000
    batch_size = 32
    patience = 200
    lr = 1e-4

    # Load data
    X, y = datasets.load('make_regression_target')

    # Define architecture to use
    model = RandomForestRegressor()

    # Split source into train and validation
    splits = train_test_split(
                              X,
                              y,
                              train_size=0.8,
                              random_state=0,
                              )
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df = pd.DataFrame()
    df['y'] = y_test
    df['y_pred'] = y_pred
    df['set'] = 'test'

    os.makedirs(save_dir, exist_ok=True)
    plots.parity(df, os.path.join(save_dir, 'parity'))


if __name__ == '__main__':
    main()
