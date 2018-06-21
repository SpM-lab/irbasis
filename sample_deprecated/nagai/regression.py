from __future__ import print_function

import numpy
import scipy
import scipy.linalg
from sklearn.linear_model import Ridge

def ridge_svd(X, y, alpha, cutoff = 1e-10):
    N1, N2 = X.shape
    U, s, Vt = scipy.linalg.svd(X, full_matrices=False)
    Nm = s.size
    idx = s > cutoff * s[0]
    s_nnz = s[idx][:, numpy.newaxis]
    UTy = numpy.dot(U.T, y)
    d = numpy.zeros((Nm,1), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d_UT_y = d.reshape((Nm,)) * UTy.reshape((Nm,))
    return numpy.dot(Vt.T, d_UT_y)

def ridge_coordinate_descent(X, y, alpha, blocks = [], rtol = 1e-8, cutoff = 1e-10):
    N1, N2 = X.shape

    x = numpy.zeros((N2,))
    r = y - numpy.dot(X, x)
    L2 = 0.0
    f = numpy.linalg.norm(r)**2 + alpha * L2

    step = 0
    while True:
        f_old = f

        for indices in blocks:
            mask = numpy.full((N2,), False, dtype=bool)
            mask[indices] = True
    
            x_A_old = numpy.array(x[indices])
    
            tilde_y = r + numpy.dot(X[:, indices], x_A_old)
            tilde_X = X[:, indices]
            x[indices] = ridge_svd(tilde_X, tilde_y, alpha, cutoff)
            r += numpy.dot(X[:, indices], x_A_old - x[indices])
            L2 += numpy.linalg.norm(x[indices])**2 - numpy.linalg.norm(x_A_old)**2
            f = numpy.linalg.norm(r)**2 + alpha * L2
    
        df = (f_old - f)/f_old
        print(step, df, numpy.linalg.norm(r)**2 + alpha * L2, numpy.linalg.norm(y - numpy.dot(X,x) - r))
    
        if df < rtol: 
            break

        step += 1
    return x


def ridge_complex(A, y, alpha, solver='svd', blocks=[]):
    (N1, N2) = A.shape
    A_big = numpy.zeros((2,N1,2,N2), dtype=float)
    A_big[0,:,0,:] =  A.real
    A_big[0,:,1,:] = -A.imag
    A_big[1,:,0,:] =  A.imag
    A_big[1,:,1,:] =  A.real

    blocks_big = []
    for b in blocks:
        blocks_big.append(numpy.concatenate([numpy.array(b), numpy.array(b)+N2], axis=0))
    
    assert len(y) == N1
    y_big = numpy.zeros((2,N1), dtype=float)
    y_big[0,:] = y.real
    y_big[1,:] = y.imag
    
    if solver == 'svd':
        coef = ridge_svd(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)), alpha)
    elif solver == 'svd_cd':
        print("calling svd_cd")
        coef = ridge_coordinate_descent(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)), alpha, blocks_big)
    elif solver == 'sparse_cg':
        #clf = Ridge(alpha=alpha, solver='sparse_cg', tol=1e-8, max_iter=1000000)
        clf = Ridge(alpha=alpha, solver='svd')
        clf.fit(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)))
        coef = clf.coef_
        print("nitr ", clf.n_iter_)
    else:
        raise RuntimeError("Uknown solver: " + solver)

    coef = coef.reshape((2,N2))
    return coef[0,:] + 1J * coef[1,:]
