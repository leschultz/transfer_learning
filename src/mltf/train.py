from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as pl
from torch import nn, optim
from models import ElemNet

import pandas as pd
import numpy as np
import joblib
import torch
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def loader(path, select, target, frac=None):

    df = pd.read_csv(path)

    if frac is not None:
        df = df.sample(frac=frac)

    X = df[select].values
    y = df[target].values

    return X, y


def save(scaler, model, df, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    torch.save(model, os.path.join(save_dir, 'model.pth'))
    df.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plot(df, os.path.join(save_dir, 'mae_vs_epochs.png'))


def plot(df, save_dir):

    valid = df[df['set'] == 'validation']

    fig, ax = pl.subplots()

    ax.plot(
            valid['epoch'],
            valid['mae'],
            marker='o',
            color='r',
            label='Validation',
            )

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss Mean Average Error')

    ax.legend()

    fig.tight_layout()
    fig.savefig(
                save_dir,
                bbox_inches='tight',
                )


def fit(
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_epochs,
        batch_size,
        lr,
        patience,
        model=None,
        ):

    # If pretrained not supplied
    if model is None:
       model = ElemNet(X_train.shape[1])

    # Define models and parameters
    scaler = StandardScaler()
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scale features
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Convert to tensor
    X_train = to_tensor(X_train)
    X_valid = to_tensor(X_valid)
    y_train = to_tensor(y_train)
    y_valid = to_tensor(y_valid)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    valid_epochs = list(range(n_epochs))
    train_epochs = []
    train_losses = []
    valid_losses = []

    best_loss = float('inf')
    no_improv = 0
    for epoch in valid_epochs:

        # Training
        model.train()
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()
            y_pred = model(X_batch)  # Foward pass
            loss = metric(y_pred, y_batch)  # Loss
            loss.backward()
            optimizer.step()

            train_epochs.append(epoch)
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            y_pred = model(X_valid)
            loss = metric(y_pred, y_valid)
            valid_losses.append(loss.item())

        if valid_losses[-1] < best_loss:
            best_loss = valid_losses[-1]
            no_improv = 0
        else:
            no_improv += 1

        if no_improv >= patience:
            break

        print(f'Epoch [{epoch+1}/{n_epochs}]')

    # Prepare data for saving
    train = pd.DataFrame()
    train['epoch'] = train_epochs
    train['mae'] = train_losses
    train['set'] = 'train'

    valid = pd.DataFrame()
    valid['epoch'] = valid_epochs
    valid['mae'] = valid_losses
    valid['set'] = 'validation'

    df = pd.concat([train, valid])

    return scaler, model, df


def to_tensor(x):
    y = torch.FloatTensor(x)

    if len(y.shape) < 2:
        y = y.reshape(-1 , 1)

    return y


def main():

    # OQMD data
    oqmd_train = '../paper/data/train_set.csv'
    oqmd_valid = '../paper/data/test_set.csv'
    oqmd_dir = '../outputs/oqmd'

    # Experimental data
    expe_train = '../paper/data/holdout-new/10/experimental_training_set.csv'
    expe_valid = '../paper/data/holdout-new/10/experimental_test_set.csv'
    expe_dir = '../outputs/expe'
    tran_dir = '../outputs/tran'

    # Parameters
    frac = None  # Can specify fraction of sub samples for fast testing
    target = 'delta_e'
    n_epochs = 1000  # Originally 1000
    batch_size = 32
    lr = 0.0001
    patience = 200
    train_size = 0.9

    # Get features
    train = pd.read_csv(oqmd_train)
    all_elements = [str(element) for element in Element]
    select = [i for i in all_elements if i in train.columns]

    # Load data for 9:1 split
    X_train_oqmd, y_train_oqmd = loader(oqmd_train, select, target, frac)
    X_valid_oqmd, y_valid_oqmd = loader(oqmd_valid, select, target, frac)

    X_train_expe, y_train_expe = loader(expe_train, select, target, frac)
    X_valid_expe, y_valid_expe = loader(expe_valid, select, target, frac)

    out = fit(
              X_train_oqmd,
              y_train_oqmd,
              X_valid_oqmd,
              y_valid_oqmd,
              n_epochs,
              batch_size,
              lr,
              patience,
              )

    oqmd_scaler, oqmd_model, oqmd_df = out
    save(*out, oqmd_dir)

    out = fit(
              X_train_expe,
              y_train_expe,
              X_valid_expe,
              y_valid_expe,
              n_epochs,
              batch_size,
              lr,
              patience,
              )

    expe_scaler, expe_model, expe_df = out
    save(*out, expe_dir)

    out = fit(
              X_train_expe,
              y_train_expe,
              X_valid_expe,
              y_valid_expe,
              n_epochs,
              batch_size,
              lr,
              patience,
              oqmd_model,
              )

    tran_scaler, tran_model, tran_df = out
    save(*out, tran_dir)


if __name__ == '__main__':
    main()
