from pandas import read_csv
import numpy as np


def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header=None)
    data.fillna('', inplace=True)
    data = data.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y.reshape(-1,1)


def train_test_split(X,Y,fraction):
    """
    perform the split of the data into training and tesing sets
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        fraction: number between 0 and 1, specifies the size of the training
                data
    """
    if fraction < 0 or fraction > 1:
        raise Exception("Fraction for split is not valid")

    np.random.seed(1)
    
    ind = int(len(Y) * fraction)
    ch_range = np.arange(len(Y))
    np.random.shuffle(ch_range)
    return X[ch_range[:ind],:], Y[ch_range[:ind]], X[ch_range[ind:],:], Y[ch_range[ind:]]
