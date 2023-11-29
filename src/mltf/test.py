from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as pl
from torch import nn, optim

import pandas as pd
import numpy as np
import joblib
import torch
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def save(scaler, model, df, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    torch.save(model, os.path.join(save_dir, 'model.pth'))
    df.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plot(df, os.path.join(save_dir, 'mae_vs_epochs.png'))


def plot(df, save_dir):

    test = df[df['set'] == 'test']

    fig, ax = pl.subplots()

    ax.plot(
            test['epoch'],
            test['mae'],
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
        X_test,
        y_test,
        n_epochs,
        batch_size,
        lr,
        patience,
        model,
        ):

    # Define models and parameters
    scaler = StandardScaler()
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scale features
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensor
    X_train = to_tensor(X_train)
    X_test = to_tensor(X_test)
    y_train = to_tensor(y_train)
    y_test = to_tensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    test_epochs = []
    train_epochs = []
    train_losses = []
    test_losses = []

    best_loss = float('inf')
    no_improv = 0
    for epoch in range(n_epochs):

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
            y_pred = model(X_test)
            loss = metric(y_pred, y_test)
            test_epochs.append(epoch)
            test_losses.append(loss.item())

        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
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

    test = pd.DataFrame()
    test['epoch'] = test_epochs
    test['mae'] = test_losses
    test['set'] = 'test'

    df = pd.concat([train, test])

    return scaler, model, df


def to_tensor(x):
    y = torch.FloatTensor(x)

    if len(y.shape) < 2:
        y = y.reshape(-1 , 1)

    return y


def run(
        X_source_train,
        y_source_train,
        X_source_test,
        y_source_test,
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        model,
        n_epochs=1000,
        batch_size=32,
        lr=0.0001,
        patience=200,
        save_dir='./outputs'
        ):

    # Fit on source domain
    out = fit(
              X_source_train,
              y_source_train,
              X_source_test,
              y_source_test,
              n_epochs,
              batch_size,
              lr,
              patience,
              copy.deepcopy(model),
              )
    source_model = out[1]
    print(out)
    save(*out, os.path.join(save_dir, 'source'))

    # Fit on target domain
    out = fit(
              X_target_train,
              y_target_train,
              X_target_test,
              y_target_test,
              n_epochs,
              batch_size,
              lr,
              patience,
              copy.deepcopy(model),
              )
    save(*out, os.path.join(save_dir, 'target'))

    # Transfer model from source to target domains
    out = fit(
              X_target_train,
              y_target_train,
              X_target_test,
              y_target_test,
              n_epochs,
              batch_size,
              lr,
              patience,
              source_model,
              )
    save(*out, os.path.join(save_dir, 'transfered'))
