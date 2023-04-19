import numpy as np
import scipy as sp
import itertools
from scipy import sparse

import obj  # for loading / saving obj


def graph_laplacian(W, normalized=True, symmetric=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        # Combinatorial Laplacian
        D = sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        if symmetric:
            # Normalized Laplacian
            d += np.spacing(np.array(0, W.dtype))
            d = 1 / np.sqrt(d)
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
        else:
            # Random-walk Laplacian
            d += np.spacing(np.array(0, W.dtype))
            d = 1.0 / d
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W

    # assert np.abs(L - L.T).mean() < 1e-9
    # assert type(L) is sparse.csr.csr_matrix
    return L


def dist(V, F):
    num_vertices = V.shape[0]
    W = np.zeros((num_vertices, num_vertices))

    for face in F:
        vertices = face.tolist()
        for i, j in itertools.product(vertices, vertices):
            W[i, j] = np.sqrt(((V[i] - V[j])**2).sum())

    return sparse.csr_matrix(W)


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
    W = sp.sparse.csr_matrix((1 / dist.data, dist.indices, dist.indptr),
                             shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    # assert np.abs(W - W.T).mean() < 1e-10
    return W


def exp_weights(dist, sigma2):
    W = sp.sparse.csr_matrix(
        (np.exp(-dist.data**2 / sigma2), dist.indices, dist.indptr),
        shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    # assert np.abs(W - W.T).mean() < 1e-10
    return W


def cotangent_weights(F, a, l):
    W = np.zeros(l.shape)
    A = np.zeros(l.shape[0])

    for f in range(F.shape[0]):
        for v_ind in itertools.permutations(F[f].tolist()):
            i, j, k = v_ind
            W[i,
              j] += (-l[i, j]**2 + l[j, k]**2 + l[k, i]**2) / (8 * a[f] + 1e-6)
            A[i] += a[f] / 3 / 4  # each face will appear 4 times

    return sp.sparse.csr_matrix(W), sp.sparse.diags(1 / (A + 1e-9), 0)


def centroids(V, F):
    C = np.zeros(F.shape)
    for i in range(F.shape[0]):
        C[i] = (V[F[i, 0]] + V[F[i, 1]] + V[F[i, 2]]) / 3
    return C


def remove_duplicates(V, F):
    uniq_V, inverse = np.unique(V, axis=0, return_inverse=True)
    new_F = inverse[F]
    return uniq_V, new_F, inverse


def average_edge_length(V, F):
    VF = V[F]
    V0, V1, V2 = VF[:, 0], VF[:, 1], VF[:, 2]

    # side lengths
    A = np.linalg.norm((V2 - V1), axis=1)
    B = np.linalg.norm((V0 - V2), axis=1)
    C = np.linalg.norm((V1 - V0), axis=1)

    return ((A + B + C).sum()) / faces.shape[0] / 3.0


def compute_face_normals(V, F):
    Ft = np.transpose(F)
    Vt = np.transpose(V)

    VV = [
        Vt.take(Ft[0], axis=1),
        Vt.take(Ft[1], axis=1),
        Vt.take(Ft[2], axis=1)
    ]

    c = np.cross(VV[1] - VV[0], VV[2] - VV[0], axisa=0, axisb=0)
    normals = c / np.linalg.norm(c, axis=0)

    return normals


def safe_arccos(x):
    return np.arccos(x.clip(min=-1, max=1))


def compute_vertex_normals(V, F, F_normals):
    s0 = F_normals.shape[0]
    if s0 != 3:
        F_normals = np.transpose(F_normals)

    Ft = np.transpose(F)
    Vt = np.transpose(V)
    V_normals = np.zeros_like(Vt)

    VV = [
        Vt.take(Ft[0], axis=1),
        Vt.take(Ft[1], axis=1),
        Vt.take(Ft[2], axis=1)
    ]

    for i in range(3):
        d0 = VV[(i + 1) % 3] - VV[i]
        d0 = d0 / np.linalg.norm(d0)
        d1 = VV[(i + 2) % 3] - VV[i]
        d1 = d1 / np.linalg.norm(d1)
        face_angle = safe_arccos(np.sum(d0 * d1, axis=0))
        nn = F_normals * face_angle
        for j in range(3):
            V_idx = Ft[i, :]
            V_normals[j, V_idx[:]] = V_normals[j, V_idx[:]] + nn[j, :]

    return np.transpose(V_normals / np.linalg.norm(V_normals, axis=0), (0, 1))


if __name__ == "__main__":
    V, F = obj.read_obj("bunny.obj")
    dists = dist(V, F)
    areas = area(F, dists)

    W, _ = cotangent_weights(F, areas, dists)
    L = graph_laplacian(W, symmetric=False)

    L = np.asarray(L.todense())
    P = np.asarray(np.matmul(L, V))
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    u, v, w = P[:, 0], P[:, 1], P[:, 2]

    F_normals = compute_face_normals(V, F)
    V_normals = compute_vertex_normals(V, F, F_normals)
