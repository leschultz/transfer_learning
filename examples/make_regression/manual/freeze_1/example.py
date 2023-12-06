from transfernet import validate, train, models, datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


def main():

    # Parameters
    save_dir = './outputs'
    prefit_model = '../create_source_model/outputs/train/source/model.pth'
    freeze_n_layers = 1  # Layers to freeze staring from first for transfer

    # Target training parameters
    target_n_epochs = 10000
    target_batch_size = 32
    target_lr = 0.0001
    target_patience = 200

    # Load data
    X_target, y_target = datasets.load('make_regression_target')

    # Define architecture to use
    model = torch.load(prefit_model)
    weights = model['weights']  # Weights
    model = models.ExampleNet(X_target.shape[1])  # Architecture

    # Split target into train and test
    splits = train_test_split(
                              X_target,
                              y_target,
                              train_size=0.8,
                              random_state=0,
                              )
    X_target_train, X_target_test, y_target_train, y_target_test = splits

    # Validate the method by having explicit test sets
    validate.run(
                 model,
                 X_target_train=X_target_train,
                 y_target_train=y_target_train,
                 X_target_test=X_target_test,
                 y_target_test=y_target_test,
                 target_n_epochs=target_n_epochs,
                 target_batch_size=target_batch_size,
                 target_lr=target_lr,
                 target_patience=target_patience,
                 save_dir=save_dir,
                 freeze_n_layers=freeze_n_layers,
                 weights=weights,
                 scratch=False,
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
              scratch=False,
              )


if __name__ == '__main__':
    main()
