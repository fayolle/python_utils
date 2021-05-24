import numpy as np
import scipy as sp
import itertools
from scipy import sparse

import obj # for loading / saving obj


def graph_laplacian(W, normalized=True, symmetric=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        if symmetric:
            d += np.spacing(np.array(0, W.dtype))
            d = 1 / np.sqrt(d)
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
        else:
            d += np.spacing(np.array(0, W.dtype))
            d = 1.0 / d
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sparse.csr.csr_matrix
    return L


def dist(V, F):
    num_vertices = V.shape[0]
    W = np.zeros((num_vertices, num_vertices))

    for face in F:
        vertices = face.tolist()
        for i, j in itertools.product(vertices, vertices):
            W[i, j] = np.sqrt(((V[i] - V[j]) ** 2).sum())

    return sparse.csr_matrix(W)


def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b,  a, -d,  c],
                     [c,  d,  a, -b],
                     [d, -c,  b,  a]])


def dirac(V, F):
    l = dist(V, F)
    Af = area(F, l)
    Av = np.zeros(V.shape[0])

    D = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
    DA = np.zeros((4 * V.shape[0], 4 * F.shape[0]))

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            Av[j] += Af[i] / 3

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            ind1 = F[i, (ind + 1) % 3]
            ind2 = F[i, (ind + 2) % 3]

            e1 = V[ind1]
            e2 = V[ind2]

            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]])

            mat = -quaternion_matrix(e) / (2 * Af[i])
            D[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat
            DA[j * 4:(j + 1) * 4, i * 4: (i + 1) * 4] = mat.transpose() * Af[i] / Av[j]

    D = sparse.csr_matrix(D)
    DA = sparse.csr_matrix(DA)

    return D, DA


def area(F, l):
    areas = np.zeros(F.shape[0])

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        sijk = (l[i, j] + l[j, k] + l[k, i]) / 2

        sum_ = sijk * (sijk - l[i, j]) * (sijk - l[j, k]) * (sijk - l[k, i])
        if sum_ > 0:
            areas[f] = np.sqrt(sum_)
        else:
            areas[f] = 1e-6

    return areas


def uniform_weights(dist):
    W = sp.sparse.csr_matrix((1 / dist.data, dist.indices, dist.indptr), shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    assert np.abs(W - W.T).mean() < 1e-10
    return W


def exp_weights(dist, sigma2):
    W = sp.sparse.csr_matrix((np.exp(-dist.data**2 / sigma2), dist.indices, dist.indptr), shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    assert np.abs(W - W.T).mean() < 1e-10
    return W


def cotangent_weights(F, a, l):
    W = np.zeros(l.shape)
    A = np.zeros(l.shape[0])

    for f in range(F.shape[0]):
        for v_ind in itertools.permutations(F[f].tolist()):
            i, j, k = v_ind
            W[i, j] += (-l[i, j]**2 + l[j, k]**2 + l[k, i]**2) / (8 * a[f] + 1e-6)
            A[i] += a[f] / 3 / 4 # each face will appear 4 times

    return sp.sparse.csr_matrix(W), sp.sparse.diags(1/(A+1e-9), 0)


def laplacian(W, A_inv):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    D = sp.sparse.diags(d.A.squeeze(), 0)
    L = A_inv * (D - W)

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.sparse.csr.csr_matrix
    return L


def adjacency_matrix_from_faces(F, num_vertices):
    A_v = np.zeros((num_vertices, num_vertices))
    A_f0 = np.zeros((num_vertices, num_vertices))
    A_f1 = np.zeros((num_vertices, num_vertices))

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        A_v[i, j] = A_v[j, i] = 1
        A_v[j, k] = A_v[k, j] = 1
        A_v[k, i] = A_v[i, k] = 1

        A_f[i][f] = 1
        A_f[j][f] = 1
        A_f[k][f] = 1

    return sp.sparse.csr_matrix(A_v), sp.sparse.csr_matrix(A_f)


def centroids(V, F):
    C = np.zeros(F.shape)
    for i in range(F.shape[0]):
        C[i] = (V[F[i, 0]] + V[F[i, 1]] + V[F[i, 2]]) / 3
    return C


if __name__ == "__main__":
    V, F = obj.read_obj("bunny.obj")
    dists = dist(V, F)
    areas = area(F, dists)

    W, _ = cotangent_weights(F, areas, dists)
    L = graph_laplacian(W, symmetric=False)
    C = centroids(V, F)

    D, DA = dirac(V, F)
    D = D.todense()
    DA = DA.todense()

    L = np.asarray(L.todense())
    P = np.asarray(np.matmul(L, V))
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    u, v, w = P[:, 0], P[:, 1], P[:, 2]

    V_ = np.pad(P, [[0, 0], [1, 0]], 'constant')
    V_ = V_.reshape(-1, 1)

    F_ = np.matmul(D, V_)

    F__ = np.matmul(DA, F_)

    F_ = np.array(F_.reshape(-1).reshape(F.shape[0], -1))
    F__ = np.array(F__.reshape(-1).reshape(V.shape[0], -1))