from transfernet.utils import validate_fit, save, freeze
import torch
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def run(
        model,
        X_source_train=None,
        y_source_train=None,
        X_source_val=None,
        y_source_val=None,
        X_target_train=None,
        y_target_train=None,
        X_target_val=None,
        y_target_val=None,
        source_n_epochs=1000,
        source_batch_size=32,
        source_lr=0.0001,
        source_patience=200,
        target_n_epochs=1000,
        target_batch_size=32,
        target_lr=0.0001,
        target_patience=200,
        freeze_n_layers=0,
        save_dir='./outputs',
        scratch=True,
        transfer=True,
        weights=None,
        scaler=None,
        ):

    cond = [
            X_source_train is None,
            y_source_train is None,
            X_source_val is None,
            y_source_val is None,
            ]
    cond = not any(cond)

    if cond:

        # Fit on source domain
        out = validate_fit(
                           X_source_train,
                           y_source_train,
                           X_source_val,
                           y_source_val,
                           source_n_epochs,
                           source_batch_size,
                           source_lr,
                           source_patience,
                           copy.deepcopy(model),
                           scaler=scaler,
                           )
        source_model = out[1]
        save(*out, save_dir=os.path.join(save_dir, 'validation/source'))

    cond = [
            X_target_train is None,
            y_target_train is None,
            X_target_val is None,
            y_target_val is None,
            ]
    cond = not any(cond)

    if scratch and cond:

        # Fit on target domain
        out = validate_fit(
                           X_target_train,
                           y_target_train,
                           X_target_val,
                           y_target_val,
                           target_n_epochs,
                           target_batch_size,
                           target_lr,
                           target_patience,
                           copy.deepcopy(model),
                           scaler=scaler,
                           )
        save(*out, save_dir=os.path.join(save_dir, 'validation/target'))

    if transfer and cond:

        if weights is not None:
            source_model = model
            source_model.load_state_dict(weights)

        # Transfer model from source to target domains
        source_model = freeze(source_model, freeze_n_layers)
        out = validate_fit(
                           X_target_train,
                           y_target_train,
                           X_target_val,
                           y_target_val,
                           target_n_epochs,
                           target_batch_size,
                           target_lr,
                           target_patience,
                           source_model,
                           scaler=scaler,
                           )
        save(*out, save_dir=os.path.join(save_dir, 'validation/transfered'))
