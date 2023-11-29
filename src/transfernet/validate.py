from transfernet.utils import validate_fit, save
import torch
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
    out = validate_fit(
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
    save(*out, os.path.join(save_dir, 'validation/source'))

    # Fit on target domain
    out = validate_fit(
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
    save(*out, os.path.join(save_dir, 'validation/target'))

    # Transfer model from source to target domains
    out = validate_fit(
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
    save(*out, os.path.join(save_dir, 'validation/transfered'))
