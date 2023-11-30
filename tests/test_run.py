from sklearn.model_selection import train_test_split
from transfernet import validate, train, models
from sklearn import datasets
import pandas as pd
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):

        # Parameters
        save_dir = './outputs'

        # Source training parameters
        source_n_epochs = 2
        source_batch_size = 32
        source_lr = 0.0001
        source_patience = 200

        # Target training parameters
        target_n_epochs = 2
        target_batch_size = 32
        target_lr = 0.0001
        target_patience = 200

        X, y = datasets.make_regression(
                                        n_samples=1000,
                                        n_features=5,
                                        random_state=0,
                                        n_targets=1,
                                        )

        # Define architecture to use
        model = models.ExampleNet(X.shape[1])

        # Split on data we start from (source) to what we transfer to (target)
        splits = train_test_split(X, y, train_size=0.8, random_state=0)
        X_source, X_target, y_source, y_target = splits

        # Make the target related to the source target by simple function
        y_target = 5*y_target+2

        # Split source into train and test
        splits = train_test_split(
                                  X_source,
                                  y_source,
                                  train_size=0.8,
                                  random_state=0,
                                  )
        X_source_train, X_source_test, y_source_train, y_source_test = splits

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
                     X_source_train,
                     y_source_train,
                     X_source_test,
                     y_source_test,
                     X_target_train,
                     y_target_train,
                     X_target_test,
                     y_target_test,
                     model,
                     source_n_epochs,
                     source_batch_size,
                     source_lr,
                     source_patience,
                     target_n_epochs,
                     target_batch_size,
                     target_lr,
                     target_patience,
                     save_dir,
                     )

        # Train 1 model on all data
        train.run(
                  X_source,
                  y_source,
                  X_target,
                  y_target,
                  model,
                  source_n_epochs,
                  source_batch_size,
                  source_lr,
                  source_patience,
                  target_n_epochs,
                  target_batch_size,
                  target_lr,
                  target_patience,
                  save_dir,
                  )

        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()
