import base64
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

def get_pdf(file_path):
    """
    Get the pdf file from the file path. Used to display the documentation in our web app.
    :param file_path:
    :return:
    """
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0" ' \
                  f'width="100%" height="1000" frameborder="0"></iframe>'
    return pdf_display


def get_indice_sets_stations(b_sol):
    # make sure that b_sol is an integer array
    if b_sol.dtype != 'int':
        raise ValueError('b_sol is not binary')

    return np.argwhere(b_sol == 1).flatten(), np.argwhere(b_sol == 0).flatten()


def get_distance_matrix(locations_1, locations_2):
    """
    Get the distance matrix with (m, k) and (n, k).
    :param locations_1: (m, k) array
    :param locations_2: (n, k) array
    :return: (m, n) distance matrix
    """
    return cdist(locations_1, locations_2)


def geometric_median(X, eps=1e-5):
    # source: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def compute_maximum_matching(n: np.ndarray, reachable: np.ndarray):
    graph = np.repeat(reachable, 2 * n, axis=1)
    result = maximum_bipartite_matching(csr_matrix(graph), perm_type='column')

    return np.mean(result >= 0)
