from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
         X_val=None,
         y_val=None,
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

    df.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plot(df, os.path.join(save_dir, 'mae_vs_epochs'))
    np.savetxt(os.path.join(save_dir, 'X_train.csv'), X_train, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'y_train.csv'), y_train, delimiter=',')

    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    if X_val is not None:
        np.savetxt(
                   os.path.join(save_dir, 'X_validation.csv'),
                   X_val,
                   delimiter=',',
                   )

    if y_val is not None:
        np.savetxt(
                   os.path.join(save_dir, 'y_validation.csv'),
                   y_val,
                   delimiter=',',
                   )


def plot(df, save_dir):

    for group, values in df.groupby('set'):

        if group == 'train':
            color = 'b'
        elif group == 'validation':
            color = 'r'

        # Regular plot
        fig, ax = pl.subplots()

        x = values['epoch'].values
        y = values['mae'].values

        val = np.min(y)

        label = '{}: lowest MAE value: {:.2f}'.format(group.capitalize(), val)
        label += '\n'
        label += '{}: last MAE value: {:.2f}'.format(group.capitalize(), y[-1])

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


def fit(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        n_epochs=1000,
        batch_size=32,
        lr=1e-4,
        patience=100,
        print_n=100,
        scaler=None,
        save_dir=None,
        ):

    valcond = all([X_val is not None, y_val is not None])

    print('_'*79)
    if valcond:
        print('Assessing model with validation set')
    else:
        print('Training the model')
    print('-'*79)

    # Define models and parameters
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scale features
    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        if valcond:
            X_val = scaler.transform(X_val)

    # Convert to tensor
    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)

    train_epochs = []
    train_losses = []

    if valcond:
        X_val = to_tensor(X_val)
        y_val = to_tensor(y_val)

        val_epochs = []
        val_losses = []

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

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
        loss = loss.item()
        train_epochs.append(epoch)
        train_losses.append(loss)

        if valcond:

            model.eval()
            with torch.no_grad():
                y_pred = model(X_val)
                loss = metric(y_pred, y_val)
                loss = loss.item()
                val_epochs.append(epoch)
                val_losses.append(loss)

        if valcond and (val_losses[-1] < best_loss):
            best_loss = val_losses[-1]
            no_improv = 0

        elif train_losses[-1] < best_loss:
            best_loss = train_losses[-1]
            no_improv = 0

        else:
            no_improv += 1

        if no_improv >= patience:
            break

        npoch = epoch+1
        if npoch % print_n == 0:
            if valcond:
                print(f'Epoch {npoch}/{n_epochs}: Validation loss {loss:.2f}')
            else:
                print(f'Epoch {npoch}/{n_epochs}: Train loss {loss:.2f}')

    # Prepare data for saving
    df = pd.DataFrame()
    df['epoch'] = train_epochs
    df['mae'] = train_losses
    df['set'] = 'train'

    if valcond:
        val = pd.DataFrame()
        val['epoch'] = val_epochs
        val['mae'] = val_losses
        val['set'] = 'validation'
        df = pd.concat([df, val])

    if save_dir is not None:
        save(
             scaler,
             model,
             df,
             X_train,
             y_train,
             X_val,
             y_val,
             save_dir=save_dir,
             )

    return scaler, model, df, X_train, y_train, X_val, y_val
