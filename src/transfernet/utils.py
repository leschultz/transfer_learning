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
import json
import copy
import os


def freeze(model, freeze_n_layers=0):

    # Freeze neural net layers
    for i, layer in enumerate(model.named_children()):
        if i < freeze_n_layers:
            for param in layer[1].parameters():
                param.requires_grad = False

    return model


def to_tensor(x):
    y = torch.FloatTensor(x)

    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    return y


def save(
         scaler,
         model,
         df,
         X_train,
         y_train,
         X_test=None,
         y_test=None,
         save_dir='./outputs',
         ):

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
               {
                'model': model,
                'weights': model.state_dict(),
                },
               os.path.join(save_dir, 'model.pth')
               )

    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    df.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plot(df, os.path.join(save_dir, 'mae_vs_epochs'))
    np.savetxt(os.path.join(save_dir, 'X_train.csv'), X_train, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'y_train.csv'), y_train, delimiter=',')

    if X_test is not None:
        np.savetxt(
                   os.path.join(save_dir, 'X_test.csv'),
                   X_test,
                   delimiter=',',
                   )

    if y_test is not None:
        np.savetxt(
                   os.path.join(save_dir, 'y_test.csv'),
                   y_test,
                   delimiter=',',
                   )


def plot(df, save_dir):

    for group, values in df.groupby('set'):

        if group == 'train':
            color = 'b'
        elif group == 'test':
            color = 'r'

        # Regular plot
        fig, ax = pl.subplots()
        val = min(values['mae'])
        label = '{} lowest value: {:.2f}'.format(group.capitalize(), val)
        ax.plot(
                values['epoch'],
                values['mae'],
                marker='o',
                color=color,
                label=label,
                )

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Mean Average Error')

        fig.tight_layout()
        name = save_dir+'_{}'.format(group)
        fig.savefig(
                    name+'.png',
                    bbox_inches='tight',
                    )

        # Legend by itself
        fig_legend, ax_legend = pl.subplots()
        ax_legend.axis(False)
        legend = ax_legend.legend(
                                  *ax.get_legend_handles_labels(),
                                  frameon=False,
                                  loc='center',
                                  bbox_to_anchor=(0.5, 0.5)
                                  )
        ax_legend.spines['top'].set_visible(False)
        ax_legend.spines['bottom'].set_visible(False)
        ax_legend.spines['left'].set_visible(False)
        ax_legend.spines['right'].set_visible(False)

        fig_legend.savefig(
                           name+'_legend.png',
                           bbox_inches='tight',
                           )

        data = {}
        data['mae'] = values['mae'].tolist()
        data['epoch'] = values['epoch'].tolist()

        with open(name+'.json', 'w') as handle:
            json.dump(data, handle)


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

        loss = metric(model(X_train), y_train)
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

    return scaler, model, df, X_train, y_train, X_test, y_test


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

        loss = metric(model(X), y)
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

    return scaler, model, df, X, y
