from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transfernet import models, datasets, utils
import pandas as pd
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):

        # Parameters
        save_dir = './outputs'
        freeze_n_layers = 1  # Layers to freeze staring from first for transfer

        # Source training parameters
        n_epochs = 1
        batch_size = 32
        lr = 0.0001
        patience = 200

        # Load data
        X, y = datasets.load('make_regression_source')

        # Define architecture to use
        model = models.ExampleNet()

        # Split source into train and validation
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

        # Validate the method by having explicit validation set
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
                  patience=patience,
                  save_dir=save_dir,
                  scaler=StandardScaler(),
                  )

        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()
