from transfernet.utils import train_fit, save, freeze
import torch
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def run(
        X_source,
        y_source,
        X_target,
        y_target,
        model,
        source_n_epochs=1000,
        source_batch_size=32,
        source_lr=0.0001,
        source_patience=200,
        target_n_epochs=1000,
        target_batch_size=32,
        target_lr=0.0001,
        target_patience=200,
        freeze_n_layers=0,
        save_dir='./outputs'
        ):

    # Fit on source domain
    out = train_fit(
                    X_source,
                    y_source,
                    source_n_epochs,
                    source_batch_size,
                    source_lr,
                    source_patience,
                    copy.deepcopy(model),
                    )
    source_model = out[1]
    save(*out, save_dir=os.path.join(save_dir, 'train/source'))

    # Fit on target domain
    out = train_fit(
                    X_target,
                    y_target,
                    target_n_epochs,
                    target_batch_size,
                    target_lr,
                    target_patience,
                    copy.deepcopy(model),
                    )
    save(*out, save_dir=os.path.join(save_dir, 'train/target'))

    # Transfer model from source to target domains
    source_model = freeze(source_model, freeze_n_layers)
    out = train_fit(
                    X_target,
                    y_target,
                    target_n_epochs,
                    target_batch_size,
                    target_lr,
                    target_patience,
                    source_model,
                    )
    save(*out, save_dir=os.path.join(save_dir, 'train/transfered'))
