import numpy as np
import scipy.sparse
from scipy import linalg
from scipy.sparse import issparse, csr_matrix


def graph_laplacian(W, normalized=True, symmetric=True):
    """Return the Laplacian of the weigth matrix."""
    
    # Degree vector
    d = W.sum(axis=0)

    # Laplacian matrix
    if not normalized:
        # Combinatorial Laplacian 
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        if symmetric:
            # Normalized Laplacian
            d += np.spacing(np.array(0, W.dtype)) # d += epsilon
            d = 1.0 / np.sqrt(d)
            D = scipy.sparse.diags(d.A.squeeze(), 0)
            I = scipy.sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
        else:
            # Random-walk Laplacian 
            d += np.spacing(np.array(0, W.dtype))
            d = 1.0 / d
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L = I - D * w

    return L


def graph_gradient(W):
    '''Return the (graph) gradient of the weight matrix W'''
    
    W = W.todense()
    n = W.shape[0]
    Wtri = np.triu(W,1)
    r,c = np.where(Wtri>0.0)
    v = Wtri[r,c]    
    ne = len(r)
    Dr = np.arange(0,ne); Dr = np.concatenate([Dr,Dr])
    Dc = np.zeros([2*ne], dtype='int32')
    Dc[:ne] = r
    Dc[ne:2*ne] = c
    Dv = np.zeros([2*ne])
    Dv[:ne] = np.sqrt(v)
    Dv[ne:2*ne] = -np.sqrt(v)
    G = scipy.sparse.csr_matrix((Dv, (Dr, Dc)), shape=(ne, n), dtype='float32')

    return G


def csr_row_norms(X):
    if X.dtype != np.float32:
        X = X.astype(np.float64)
    return _csr_row_norms(X.data, X.shape, X.indices, X.indptr)


def _csr_row_norms(X_data,shape,X_indices,X_indptr):
    n_samples = shape[0]
    n_features = shape[1]
    norms = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        sum_ = 0.0
        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += X_data[j] * X_data[j]
        norms[i] = sum_

    return norms


def row_norms(X, squared=False):
    if issparse(X):
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms


def safe_sparse_dot(a, b, dense_output=False):
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def euclidean_distances(X, Y=None):
    XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:
        YY = XX.T
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)


def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=1, **kwds):
    if Y is None:
        Y = X    
    return euclidean_distances(X, Y, **kwds)


def construct_knn_graph(X,k):
    '''Return the k-nearest neighbor graph'''
    
    n = X.shape[0]
    
    if isinstance(X, np.ndarray)==False:
        X = X.toarray()        
    Xzc = X - np.mean(X,axis=0)
    D = pairwise_distances(X, metric='euclidean', n_jobs=1)

    # indices of k nearest neighbors
    idx = np.argsort(D)[:,:k] 
    D.sort()
    D = D[:,:k]

    # weight matrix
    sigma2 = np.mean(D[:,-1])**2
    W = np.exp(- D**2 / sigma2)
    n = X.shape[0]
    row = np.arange(0, n).repeat(k)
    col = idx.reshape(n*k)
    data = W.reshape(n*k)
    W = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
        
    return W


def sort_pca(lamb, U):
    # Sort lamb values in decreasing order
    idx = lamb.argsort()[::-1]
    return lamb[idx], U[:,idx]


def compute_pca(X,nb_pca):
    '''
    Compute the principal components of an unstructured point-cloud X. 
    
    Use: 
    [PC,PD,EnPD] = compute_pca(X,nb_pca)
    
    Input:
    X = Data matrix. Size = n x d.
    nb_pca = Number of principal components. 
    
    Output:
    PC = Principal components. Size = n x nb_pca.
    PD = Principal directions. Size = d x nb_pca.
    EnPD = Energy/variance of the principal directions. Size = np_pca x 1.
    '''    
    Xzc = X - np.mean(X,axis=0) # zero-centered data
    
    n,d = X.shape
    if n>d:
        CovX = (Xzc.T).dot(Xzc) 
        dCovX = CovX.shape[0]

        if nb_pca<dCovX:
            # U = d x nb_pca
            lamb, U = scipy.sparse.linalg.eigsh(CovX, k=nb_pca, which='LM') 
            lamb, U = sort_pca(lamb, U)  
        else: 
            lamb, U = np.linalg.eig(CovX)
            lamb, U = sort_pca(lamb, U)        
        PD = U
        PC = Xzc.dot(U)
        EnPD = lamb
        
    else:
        U,S,V = scipy.sparse.linalg.svds(Xzc, k=nb_pca, which='LM')
        U = U[:,::-1]
        S = S[::-1]
        V = V[::-1,:]
        PD = V
        PC = U.dot(np.diag(S))
        EnPD = S**2

    return PC,PD,EnPD

