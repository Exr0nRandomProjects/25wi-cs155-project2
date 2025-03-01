from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
import pandas as pd



if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')

    # A reader is still needed but only the rating_scale param is required.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["User ID", "Movie ID", "Rating"]], reader)

    clf = SVD()
    clf.fit(data.build_full_trainset())
    cross_validate(clf, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

'''
                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.9406  0.9338  0.9374  0.9365  0.9355  0.9368  0.0023  
MAE (testset)     0.7416  0.7381  0.7394  0.7377  0.7370  0.7387  0.0016  
Fit time          0.47    0.46    0.51    0.47    0.47    0.48    0.02    
Test time         0.07    0.07    0.07    0.05    0.07    0.06    0.01 
'''