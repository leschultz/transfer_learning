from transfernet import validate, train, models, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main():

    # Parameters
    save_dir = './outputs'

    # Source training parameters
    source_n_epochs = 1000
    source_batch_size = 32
    source_lr = 0.0001
    source_patience = 200

    # Load data
    X_source, y_source = datasets.load('make_regression_source')

    # Define architecture to use
    model = models.ExampleNet(X_source.shape[1])

    # Split source into train and validation
    splits = train_test_split(
                              X_source,
                              y_source,
                              train_size=0.8,
                              random_state=0,
                              )
    X_source_train, X_source_val, y_source_train, y_source_val = splits

    # Validate the method by having explicit validation sets
    validate.run(
                 model,
                 X_source_train,
                 y_source_train,
                 X_source_val,
                 y_source_val,
                 source_n_epochs=source_n_epochs,
                 source_batch_size=source_batch_size,
                 source_lr=source_lr,
                 source_patience=source_patience,
                 save_dir=save_dir,
                 scaler=StandardScaler(),
                 )

    # Train 1 model on all data
    train.run(
              model,
              X_source,
              y_source,
              source_n_epochs=source_n_epochs,
              source_batch_size=source_batch_size,
              source_lr=source_lr,
              source_patience=source_patience,
              save_dir=save_dir,
              scaler=StandardScaler(),
              )


if __name__ == '__main__':
    main()
