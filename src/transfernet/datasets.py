from pymatgen.core import Element, Composition
import pkg_resources
import pandas as pd
import numpy as np
import os

data_path = pkg_resources.resource_filename('transfernet', 'data')


def drop_constant_cols(X):

    X = X[:, ~(X == X[0, :]).all(0)]

    return X


def features(comps):

    X = np.zeros((len(comps), 118))
    count = 0
    for comp in comps:
        comp = Composition(comp).fractional_composition

        for i, j in comp.items():
            i = i.Z-1
            X[count, i] = j

        count += 1

    return X


def load(name, frac=1, drop_constant=False):

    newname = os.path.join(data_path, name+'.csv')

    if 'make_regression' in name:
        df = pd.read_csv(newname)
        df = df.sample(frac=frac)

        y = df['y'].values
        X = df.drop(['y'], axis=1).values

    else:
        df = pd.read_csv(newname)
        df = df.sample(frac=frac)
        df = df.values

        X = features(df[:, 0])
        y = df[:, 1].astype(np.float64)

    if drop_constant:
        X = drop_constant_cols(X)

    return X, y
