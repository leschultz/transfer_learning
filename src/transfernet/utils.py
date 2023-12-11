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
         df_loss,
         X_train,
         y_train,
         X_val=None,
         y_val=None,
         save_dir='./outputs',
         ):

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
               model,
               os.path.join(save_dir, 'model.pth')
               )

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
                   X_val,
                   delimiter=',',
                   )

    if y_val is not None:
        np.savetxt(
                   os.path.join(save_dir, 'y_validation.csv'),
                   y_val,
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
        ):

    # Conditions in case validation and/or test sets are supplied
    valcond = all([X_val is not None, y_val is not None])
    testcond = all([X_test is not None, y_test is not None])

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
    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)

    train_epochs = []
    train_losses = []

    if valcond:
        X_val = to_tensor(X_val)
        y_val = to_tensor(y_val)

        val_epochs = []
        val_losses = []

    if testcond:
        X_test = to_tensor(X_test)
        y_test = to_tensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    # Variables for early stopping
    if patience is not None:
        best_loss = float('inf')
        no_improv = 0
        best_model = model

    # Training loop
    for epoch in range(n_epochs):

        model.train()  # Traning mode

        # Train for batches
        for X_batch, y_batch in train_loader:

            y_pred = model(X_batch)  # Foward pass
            loss = metric(y_pred, y_batch)  # Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # Evaluation mode

        # Predictions on training
        y_pred_train = model(X_train)
        loss = metric(y_pred_train, y_train)
        loss = loss.item()
        train_epochs.append(epoch)
        train_losses.append(loss)

        # Predictions on validation
        if valcond:

            with torch.no_grad():
                y_pred_val = model(X_val)
                loss = metric(y_pred_val, y_val)
                loss = loss.item()
                val_epochs.append(epoch)
                val_losses.append(loss)

        # If data does not improve in patience epochs
        if patience is not None:
            if valcond and (val_losses[-1] < best_loss):
                best_loss = val_losses[-1]
                no_improv = 0
                best_model = copy.deepcopy(model)

            elif train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                no_improv = 0
                best_model = copy.deepcopy(model)

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

    # Select for lowet MAE model
    if patience is not None:
        model = best_model

    # Prepare data for saving
    y_pred_train = model(X_train)
    df = pd.DataFrame()
    df['y'] = y_train.detach().view(-1)
    df['y_pred'] = y_pred_train.detach().view(-1)
    df['set'] = 'train'

    df_loss = pd.DataFrame()
    df_loss['epoch'] = train_epochs
    df_loss['mae'] = train_losses
    df_loss['set'] = 'train'

    # Aggregate validation data
    if valcond:
        y_pred_val = model(X_val)
        val = pd.DataFrame()
        val['y'] = y_val.detach().view(-1)
        val['y_pred'] = y_pred_val.detach().view(-1)
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
        test = pd.DataFrame()
        test['y'] = y_test.detach().view(-1)
        test['y_pred'] = y_pred_test.detach().view(-1)
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
             save_dir=save_dir,
             )

    return scaler, model, df, df_loss, X_train, y_train, X_val, y_val
