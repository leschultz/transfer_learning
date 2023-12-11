from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transfernet import models, datasets, utils
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

    hidden_layers = 3

    # Load data
    X, y = datasets.load('make_regression_target')

    # Define architecture to use
    model = torch.load(prefit_model)

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

    # Freeze n layers
    for n in range(hidden_layers+1):

        print('+'*79)
        print(f'Freezing {n} layers')

        # Validate the method by having explicit test set
        utils.fit(
                  model,  # Copy to start from original model
                  X_train,
                  y_train,
                  X_val=X_val,
                  y_val=y_val,
                  X_test=X_test,
                  y_test=y_test,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  lr=lr,
                  save_dir=save_dir+'/freeze_{}'.format(n),
                  scaler=StandardScaler(),
                  patience=patience,
                  freeze_n_layers=n,
                  )


if __name__ == '__main__':
    main()
