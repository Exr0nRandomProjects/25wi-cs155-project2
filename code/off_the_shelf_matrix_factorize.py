from sklearn.decomposition import NMF, DictionaryLearning
import scipy
import pandas as pd
import numpy as np

def make_sparse_X(data):
    # data['Rating'] = data['Rating'] / 5
    return scipy.sparse.coo_matrix((data['Rating'], (data['Movie ID'], data['User ID']))).T

def fit_matrix_factorization(X):
    clf = NMF()
    clf.fit(X)


if __name__ == '__main__':
    data = pd.read_csv('../data/train.csv')
    train_X = make_sparse_X(data)

    val_X = make_sparse_X(pd.read_csv('../data/test.csv'))

    # fit 
    print('fitting nmf...')
    clf = NMF(verbose=1, n_components=20, alpha_W=1e-3, alpha_H='same', shuffle=True) # W reg needed for convergence
    # clf = DictionaryLearning(verbose=1, n_components=20)
    clf.fit(train_X)
    print(clf.__dict__)
    V = clf.components_.T
    np.save('sklearn-nmf', V)
    print(V.shape)
    print('done')

    # eval
    
    # pred = clf.transform(val_X)
    # print(pred.shape)
