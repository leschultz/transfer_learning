import pkg_resources
import pandas as pd
import os

data_path = pkg_resources.resource_filename('transfernet', 'data')


def load(name):
    name = os.path.join(data_path, name+'.csv')
    df = pd.read_csv(name)

    source = df[df['set'] == 'source']
    target = df[df['set'] == 'target']

    y_source = source['y'].values
    y_target = target['y'].values

    X_source = source.drop(['y', 'set'], axis=1).values
    X_target = target.drop(['y', 'set'], axis=1).values

    return X_source, y_source, X_target, y_target
