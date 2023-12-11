from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transfernet import models, datasets, utils
import pandas as pd
import torch


def main():

    # Parameters
    save_dir = './outputs'
    prefit_model = '../create_source_model/outputs/train/model.pth'

    # Target training parameters
    n_epochs = 10000
    batch_size = 32
    lr = 0.0001
    patience=200

    # Load data
    X, y = datasets.load('make_regression_target')

    # Define architecture to use
    prefit = torch.load(prefit_model)
    new = models.ExampleNet()
    model = models.AppendModel(prefit, new)

    # Split target into train and validation
    splits = train_test_split(
                              X,
                              y,
                              train_size=0.8,
                              random_state=0,
                              )
    X_train, X_val, y_train, y_val = splits

    # Split validation to get test set
    splits = train_test_split(
                              X_val,
                              y_val,
                              train_size=0.5,
                              random_state=0,
                              )
    X_val, X_test, y_val, y_test = splits

    # Validate the method by having explicit test set
    utils.fit(
              model,
              X_train,
              y_train,
              X_val=X_val,
              y_val=y_val,
              X_test=X_test,
              y_test=y_test,
              n_epochs=n_epochs,
              batch_size=batch_size,
              lr=lr,
              save_dir=save_dir,
              scaler=StandardScaler(),
              patience=200,
              )


if __name__ == '__main__':
    main()
