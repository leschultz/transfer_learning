from transfernet import validate, train, models, datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


def main():

    # Parameters
    save_dir = './outputs'
    prefit_model = 'replace_source_model'
    freeze_n_layers = replace_freeze  # Layers to freeze staring from first for transfer

    # Target training parameters
    target_n_epochs = 1000
    target_batch_size = 32
    target_lr = 0.0001
    target_patience = 200

    # Load data
    X_target, y_target = datasets.load('replace_data')

    # Define architecture to use
    model = torch.load(prefit_model)
    weights = model['weights']  # Weights
    model = models.ExampleNet(X_target.shape[1])  # Architecture

    # Split target into train and validation
    splits = train_test_split(
                              X_target,
                              y_target,
                              train_size=0.8,
                              random_state=0,
                              )
    X_target_train, X_target_val, y_target_train, y_target_val = splits

    # Validate the method by having explicit validation sets
    validate.run(
                 model,
                 X_target_train=X_target_train,
                 y_target_train=y_target_train,
                 X_target_val=X_target_val,
                 y_target_val=y_target_val,
                 target_n_epochs=target_n_epochs,
                 target_batch_size=target_batch_size,
                 target_lr=target_lr,
                 target_patience=target_patience,
                 save_dir=save_dir,
                 freeze_n_layers=freeze_n_layers,
                 weights=weights,
                 )

    # Train 1 model on all data
    train.run(
              model,
              X_target=X_target,
              y_target=y_target,
              target_n_epochs=target_n_epochs,
              target_batch_size=target_batch_size,
              target_lr=target_lr,
              target_patience=target_patience,
              save_dir=save_dir,
              freeze_n_layers=freeze_n_layers,
              weights=weights,
              )


if __name__ == '__main__':
    main()
