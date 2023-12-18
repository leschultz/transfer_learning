from pymatgen.core import Element, Composition
import pkg_resources
import pandas as pd
import numpy as np
import os

# Added datasets
paths = {}
sets = [
        'data/synthetic',
        'data/external/zenodo_5533023',
        'data/external/oqmd',
        ]

for p in sets:
    paths[p] = pkg_resources.resource_filename('transfernet', p)


def drop_constant_cols(X):

    X = X[:, ~(X == X[0, :]).all(0)]

    return X


def features(comps):

    # There are 118 known elements
    X = np.zeros((len(comps), 118))
    count = 0
    for comp in comps:
        comp = Composition(comp).fractional_composition

        for i, j in comp.items():
            i = i.Z-1
            X[count, i] = j

        count += 1

    return X


def load(name, frac=None, drop_constant=False, featurize=True):

    if 'make_regression' in name:

        newname = paths['data/synthetic']
        newname = os.path.join(newname, name+'.csv')

        df = pd.read_csv(newname)

        if frac is not None:
            df = df.sample(frac=frac)

        y = df['y'].values
        X = df.drop(['y'], axis=1).values

    elif 'oqmd' in name:

        newname = paths['data/external/oqmd']
        newname = os.path.join(newname, name+'.csv')

        df = pd.read_csv(newname)

        if frac is not None:
            df = df.sample(frac=frac)

        df = df.values

        X = features(df[:, 0]) if featurize else df[:, 0]
        y = df[:, 1].astype(np.float64)

    else:

        newname = paths['data/external/zenodo_5533023']
        newname = os.path.join(newname, name+'.csv')

        df = pd.read_csv(newname)

        if frac is not None:
            df = df.sample(frac=frac)

        df = df.values

        X = features(df[:, 0]) if featurize else df[:, 0]
        y = df[:, 1].astype(np.float64)

    if drop_constant:
        X = drop_constant_cols(X)

    return X, y
