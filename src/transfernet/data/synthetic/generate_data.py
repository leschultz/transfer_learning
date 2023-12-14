from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd


def main():

    # Make regression
    X, y = datasets.make_regression(
                                    n_samples=1000,
                                    n_features=5,
                                    random_state=0,
                                    n_targets=1,
                                    )

    df = pd.DataFrame(X)
    df['y'] = y

    # Split on data we start from (source) to what we transfer to (target)
    splits = train_test_split(df, train_size=0.8, random_state=0)
    df_source, df_target = splits

    # Make the target related to the source target by simple function
    df_target['y'] = df_target['y'].apply(lambda x: 5*x+2)

    df_source.to_csv('make_regression_source.csv', index=False)
    df_target.to_csv('make_regression_target.csv', index=False)


if __name__ == '__main__':
    main()
