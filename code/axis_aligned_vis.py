
def make_axis_aligned_V(V, *axes):
    import pandas as pd
    import numpy as np
    movies = pd.read_csv('../data/movies.csv')
    # find centroid of latent representations of movies then project V down onto that
    cols = []
    directions = []
    for axis in axes:
        m_ids_p = (movies[movies[axis] == 1]['Movie ID']-1).to_list()
        m_ids_n = (movies[movies[axis] == 0]['Movie ID']-1).to_list()
    
        centroid_p = V[m_ids_p].mean(axis=0)
        centroid_n = V[m_ids_n].mean(axis=0)

        direction = (centroid_p - centroid_n)
        print(f"norm of direction {axis}: {np.linalg.norm(direction):4g} with mean residuals {np.linalg.norm(V[m_ids_p] - centroid_p, axis=1).mean(axis=0):.3g}, {np.linalg.norm(V[m_ids_n] - centroid_n, axis=1).mean(axis=0):.3g}")
        direction /= np.linalg.norm(direction)
        directions.append(direction)

        proj = np.dot(V, direction)
        cols.append(proj)
    
    for i, (n1, d1) in enumerate(zip(axes, directions)):
        for n2, d2 in zip(axes[i+1:], directions[i+1:]):
            print(f'angle between {n1} and {n2} = {np.acos(np.dot(d1, d2))/np.pi*180:.4g}deg')
            

    return np.vstack(cols).T


# testing
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    movies = pd.read_csv('../data/movies.csv')
    V = np.load("sklearn-nmf.npy")
    V = make_axis_aligned_V(V, movies, 'Comedy', 'Childrens')
    print('final V shape', V.shape)