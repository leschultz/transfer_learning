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


def stats(df, cols, drop=None):
    '''
    Get the statistic of a dataframe.
    '''

    if drop:
        df = df.drop(drop, axis=1)

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)
    df = df.reset_index()

    return df


def to_tensor(x):
    y = torch.FloatTensor(x)

    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    return y


def save(scaler, model, df, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    torch.save(model, os.path.join(save_dir, 'model.pth'))
    df.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plot(df, os.path.join(save_dir, 'mae_vs_epochs'))


def plot(df, save_dir):

    df = stats(df, ['set', 'epoch'])

    for group, values in df.groupby('set'):

        if group == 'train':
            color = 'b'
        elif group == 'test':
            color = 'r'

        fig, ax = pl.subplots()
        ax.errorbar(
                    values['epoch'],
                    values['mae_mean'],
                    yerr=values['mae_sem'],
                    fmt='-o',
                    color=color,
                    label=group.capitalize(),
                    )

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Mean Average Error')

        ax.legend()

        fig.tight_layout()
        fig.savefig(
                    save_dir+'_{}.png'.format(group),
                    bbox_inches='tight',
                    )


def validate_fit(
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


def train_fit(
              X,
              y,
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
    scaler.fit(X)
    X = scaler.transform(X)

    # Convert to tensor
    X = to_tensor(X)
    y = to_tensor(y)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    train_epochs = []
    train_losses = []

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

        if train_losses[-1] < best_loss:
            best_loss = train_losses[-1]
            no_improv = 0
        else:
            no_improv += 1

        if no_improv >= patience:
            break

        print(f'Epoch [{epoch+1}/{n_epochs}]')

    # Prepare data for saving
    df = pd.DataFrame()
    df['epoch'] = train_epochs
    df['mae'] = train_losses
    df['set'] = 'train'

    return scaler, model, df
