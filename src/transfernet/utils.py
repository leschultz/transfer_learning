from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch import nn, optim
from transfernet import plots

import pandas as pd
import numpy as np
import joblib
import torch
import copy
import json
import os

# Chose defalut device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def freeze(model, freeze_n_layers=0):

    # Custom support for custom model
    if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
        layers = model.layers.named_children()
    # More general way people build models
    else:
        layers = model.named_children()

    # Freeze neural net layers
    for i, layer in enumerate(layers):
        if i < freeze_n_layers:
            for param in layer[1].parameters():
                param.requires_grad = False

    return model


def to_tensor(x, device):
    y = torch.FloatTensor(x).to(device)

    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    return y


def save(
         scaler,
         model,
         df,
         df_loss,
         X_train,
         y_train,
         X_val=None,
         y_val=None,
         X_test=None,
         y_test=None,
         save_dir='./outputs',
         ):

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
               model,
               os.path.join(save_dir, 'model.pth')
               )

    X_train = X_train.cpu().detach()
    df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    df_loss.to_csv(os.path.join(save_dir, 'mae_vs_epochs.csv'), index=False)
    plots.learning_curve(df_loss, os.path.join(save_dir, 'mae_vs_epochs'))
    plots.parity(df, os.path.join(save_dir, 'parity'))
    np.savetxt(os.path.join(save_dir, 'X_train.csv'), X_train, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'y_train.csv'), y_train, delimiter=',')

    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    if X_val is not None:
        np.savetxt(
                   os.path.join(save_dir, 'X_validation.csv'),
                   X_val.cpu().detach(),
                   delimiter=',',
                   )

    if y_val is not None:
        np.savetxt(
                   os.path.join(save_dir, 'y_validation.csv'),
                   y_val,
                   delimiter=',',
                   )

    if X_test is not None:
        np.savetxt(
                   os.path.join(save_dir, 'X_test.csv'),
                   X_test.cpu().detach(),
                   delimiter=',',
                   )

    if y_test is not None:
        np.savetxt(
                   os.path.join(save_dir, 'y_test.csv'),
                   y_test,
                   delimiter=',',
                   )


def fit(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        n_epochs=1000,
        batch_size=32,
        lr=1e-4,
        patience=None,
        print_n=100,
        scaler=None,
        save_dir=None,
        freeze_n_layers=0,
        pick='lowest',
        device=device,
        ):

    # Conditions in case validation and/or test sets are supplied
    valcond = all([X_val is not None, y_val is not None])  # Val set
    testcond = all([X_test is not None, y_test is not None])  # Test set
    patcond = patience is not None  # Patience

    model = copy.deepcopy(model).to(device)

    print('_'*79)
    if valcond:
        print('Assessing model with validation set')
    else:
        print('Training the model')
    print('-'*79)

    # The number of layers to freeze (useful if pretrained model supplied)
    freeze(model, freeze_n_layers)

    # Define models and parameters
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scale features
    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        if valcond:
            X_val = scaler.transform(X_val)

        if testcond:
            X_test = scaler.transform(X_test)

    # Convert to tensor
    X_train = to_tensor(X_train, device)
    y_train = to_tensor(y_train, device)

    train_epochs = []
    train_losses = []

    if valcond:
        X_val = to_tensor(X_val, device)
        y_val = to_tensor(y_val, device)

        val_epochs = []
        val_losses = []

    if testcond:
        X_test = to_tensor(X_test, device)
        y_test = to_tensor(y_test, device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    # Training loop
    no_improv = 0
    best_loss = float('inf')
    for epoch in range(n_epochs):

        model.train()  # Traning model

        # Train for batches
        for X_batch, y_batch in train_loader:

            y_pred = model(X_batch)  # Foward pass
            loss = metric(y_pred, y_batch)  # Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()  # Evaluation model

            # Predictions on training
            y_pred_train = model(X_train)
            loss = metric(y_pred_train, y_train)
            loss = loss.item()
            train_epochs.append(epoch)
            train_losses.append(loss)

            # Predictions on validation
            if valcond:
                y_pred_val = model(X_val)
                loss = metric(y_pred_val, y_val)
                loss = loss.item()
                val_epochs.append(epoch)
                val_losses.append(loss)

            # Check for lowest loss
            if valcond and (val_losses[-1] < best_loss):
                best_model = copy.deepcopy(model)
                best_loss = val_losses[-1]
                no_improv = 0

            elif train_losses[-1] < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = train_losses[-1]
                no_improv = 0

            else:
                no_improv += 1

            # If data does not improve in patience epochs
            if patcond:
                if no_improv >= patience:
                    break

            npoch = epoch+1
            if npoch % print_n == 0:
                p = f'Epoch {npoch}/{n_epochs}: '
                if valcond:
                    print(p+f'Validation loss {loss:.2f}')
                else:
                    print(p+f'Train loss {loss:.2f}')

    # Select for lowest MAE model
    if pick == 'lowest':
        model = best_model

    # Prepare data for saving
    y_pred_train = model(X_train)

    y_train = y_train.cpu().detach().view(-1)
    y_pred_train = y_pred_train.cpu().detach().view(-1)

    df = pd.DataFrame()
    df['y'] = y_train
    df['y_pred'] = y_pred_train
    df['set'] = 'train'

    df_loss = pd.DataFrame()
    df_loss['epoch'] = train_epochs
    df_loss['mae'] = train_losses
    df_loss['set'] = 'train'

    # Aggregate validation data
    if valcond:
        y_pred_val = model(X_val)

        y_val = y_val.cpu().detach().view(-1)
        y_pred_val = y_pred_val.cpu().detach().view(-1)

        val = pd.DataFrame()
        val['y'] = y_val
        val['y_pred'] = y_pred_val
        val['set'] = 'validation'

        val_loss = pd.DataFrame()
        val_loss['epoch'] = val_epochs
        val_loss['mae'] = val_losses
        val_loss['set'] = 'validation'

        df = pd.concat([df, val])
        df_loss = pd.concat([df_loss, val_loss])

    # Predictions on test and aggregate data
    if testcond:
        y_pred_test = model(X_test)

        y_test = y_test.cpu().detach().view(-1)
        y_pred_test = y_pred_test.cpu().detach().view(-1)

        test = pd.DataFrame()
        test['y'] = y_test
        test['y_pred'] = y_pred_test
        test['set'] = 'test'

        df = pd.concat([df, test])

    # Save all important things
    if save_dir is not None:
        save(
             scaler,
             model,
             df,
             df_loss,
             X_train,
             y_train,
             X_val,
             y_val,
             X_test,
             y_test,
             save_dir=save_dir,
             )

    return scaler, model, df, df_loss, X_train, y_train, X_val, y_val
