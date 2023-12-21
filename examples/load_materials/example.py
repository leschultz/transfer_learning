from transfernet import datasets


def main():

    # Load data
    X, y = datasets.load('oqmd_formation')

    print(X)
    print(y)

if __name__ == '__main__':
    main()
