from transfernet.utils import train_fit, save
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
        n_epochs=1000,
        batch_size=32,
        lr=0.0001,
        patience=200,
        save_dir='./outputs'
        ):

    # Fit on source domain
    out = train_fit(
                    X_source,
                    y_source,
                    n_epochs,
                    batch_size,
                    lr,
                    patience,
                    copy.deepcopy(model),
                    )
    source_model = out[1]
    save(*out, os.path.join(save_dir, 'train/source'))

    # Fit on target domain
    out = train_fit(
                    X_target,
                    y_target,
                    n_epochs,
                    batch_size,
                    lr,
                    patience,
                    copy.deepcopy(model),
                    )
    save(*out, os.path.join(save_dir, 'train/target'))

    # Transfer model from source to target domains
    out = train_fit(
                    X_target,
                    y_target,
                    n_epochs,
                    batch_size,
                    lr,
                    patience,
                    source_model,
                    )
    save(*out, os.path.join(save_dir, 'train/transfered'))
