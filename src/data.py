import math
import numpy as np
from sklearn.datasets import load_digits


def fetch_dataset():
    return load_digits(as_frame=True).frame


def test_train_split(dataset, test_perc=0.2, random_state=2025):
    # Set seed for reproducible shuffling
    np.random.seed(random_state)

    # Shuffle the dataset randomly
    shuffled_dataset = np.random.permutation(dataset)

    # Calculate split
    test_size = math.ceil(test_perc * shuffled_dataset.shape[0])

    # Split the dataset into training and test set
    X_test = shuffled_dataset[:test_size, :-1]
    y_test = shuffled_dataset[:test_size, -1:].reshape((-1,)).astype(int)

    X = shuffled_dataset[test_size, :-1]
    y = shuffled_dataset[test_size: -1, ].reshape((-1,)).astype(int)

    return X, X_test, y, y_test


if __name__ == '__main__':
    data = fetch_dataset()
    Xtr, Xte, ytr, yte = test_train_split(data)

    print(f"Training Data Shape: {Xtr.shape}")
    print(f"Training Labels Shape: {ytr.shape}")
    print(f"Test Data Shape: {Xte.shape}")
    print(f"Test Labels Shape: {yte.shape}")