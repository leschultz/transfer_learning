from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transfernet import models, datasets, utils
import pandas as pd
import torch


def main():

    # Parameters
    save_dir = './outputs'
    prefit_model = '../create_source_model/outputs/train/model.pth'
    freeze_n_layers = 0  # Layers to freeze staring from first for transfer

    # Target training parameters
    n_epochs = 10000
    batch_size = 32
    lr = 0.0001
    patience = 200

    # Load data
    X, y = datasets.load('make_regression_target')

    # Define architecture to use
    model = torch.load(prefit_model)
    weights = model['weights']  # Weights
    model = models.ExampleNet()  # Architecture

    # Split target into train and validation
    splits = train_test_split(
                              X,
                              y,
                              train_size=0.8,
                              random_state=0,
                              )
    X_train, X_val, y_train, y_val = splits

    # Validate the method by having explicit validation set
    utils.fit(
              model,
              X_train,
              y_train,
              X_val=X_val,
              y_val=y_val,
              n_epochs=n_epochs,
              batch_size=batch_size,
              lr=lr,
              patience=patience,
              save_dir=save_dir+'/validation',
              scaler=StandardScaler(),
              )

    # Train model on all data
    utils.fit(
              model,
              X,
              y,
              n_epochs=n_epochs,
              batch_size=batch_size,
              lr=lr,
              patience=patience,
              save_dir=save_dir+'/train',
              scaler=StandardScaler(),
              )


if __name__ == '__main__':
    main()
