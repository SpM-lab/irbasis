import numpy
import scipy
import scipy.linalg
from irbasis import *
from regression import *

def _my_mod(t, beta):
    t_new = t
    s = 1
    
    max_loop = 10

    loop = 0
    while t_new >= beta and loop < max_loop:
        t_new -= beta
        s *= -1
        loop += 1
        
    loop = 0
    while t_new < 0 and loop < max_loop:
        t_new += beta
        s *= -1
        loop += 1

    if not (t_new >= 0 and t_new <= beta):
        print("Error in _my_mod ", t, t_new, beta)

    assert t_new >= 0 and t_new <= beta
    
    return t_new, s

def _find_zeros(ulx):
    Nx = 10000
    eps = 1e-10
    tvec = numpy.linspace(-3, 3, Nx) #3 is a very safe option.
    xvec = numpy.tanh(0.5*numpy.pi*numpy.sinh(tvec))

    zeros = []
    for i in range(Nx-1):
        if ulx(xvec[i]) * ulx(xvec[i+1]) < 0:
            a = xvec[i+1]
            b = xvec[i]
            u_a = ulx(a)
            u_b = ulx(b)
            while a-b > eps:
                half_point = 0.5*(a+b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5*(a+b))
    return numpy.array(zeros)

class TwoPointGFBasis(object):
    def __init__(self, statistics, Lambda, beta, Nl = -1, cutoff = 1e-8):
        self._Lambda = Lambda
        self._beta = beta
        self._stat = statistics
        self._b = irbasis.load(self._stat, Lambda)
        if Nl == -1:
            self._Nl = self._b.dim()
        else:
            self._Nl = Nl

        assert self._Nl <= self._b.dim()
        self._scale = numpy.sqrt(2/self._beta)
        self._sqrt_beta = numpy.sqrt(self._beta)

        self._Tnl_cache = {}

    def dim(self):
        return self._Nl

    def projector_tau(self, tau):
        return numpy.array([self._Ultau(l, tau) for l in range(self._Nl)])

    def projector_iwn(self, n):
        return self._sqrt_beta * self._get_Tnl(n)

    def evaluate_tau(self, coeff, tau):
        M = self.projector_tau(tau)
        return numpy.dot(M.reshape((1, self._Nl)), coeff.reshape((self._Nl, 1))) [0,0]

    def evaluate_iwn(self, coeff, n):
        M = self.projector_iwn(n)
        return numpy.dot(M.reshape((1, self._Nl)), coeff.reshape((self._Nl, 1))) [0,0]

    def precompute_Tnl(self, n_vec):
        num_n = len(n_vec)
        Tnl = self._b.compute_Tnl(n_vec)[:,0:self._Nl].reshape((num_n, self._Nl))
        for i in range(num_n):
            self._Tnl_cache[n_vec[i]] = Tnl[i,:]

    def _get_Tnl(self, n):
        if not n in self._Tnl_cache:
            self._Tnl_cache[n] = self._b.compute_Tnl(numpy.array([n]))[:,0:self._Nl].reshape((self._Nl))
        return self._Tnl_cache[n]

    def _Ultau(self, l, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        assert -1-1e-10 < x and x < 1 + 1e+10
        sign = s if self._stat == 'F' else 1
        return sign * self._b.ulx(l, x) * self._scale

def fit_Giw(basis, X, Y, solver='svd'):
    assert len(X) == len(Y)
    num_data = len(X)
    Nl = basis.dim()

    basis.precompute_Tnl(X)
    A = numpy.zeros((num_data, Nl), dtype=complex)
    for i in range(num_data):
        A[i, :] = basis.projector_iwn(X[i]) * (abs(X[i]) + 1)

    Y_scaled = Y * (numpy.abs(X) + 1)

    return ridge_complex(A, Y_scaled, alpha = 1e-10, solver=solver)

def _eigh_ordered(mat):
    n=mat.shape[0]
    evals,evecs=numpy.linalg.eigh(mat)
    idx=numpy.argsort(evals)
    evecs2=numpy.zeros_like(evecs)
    evals2=numpy.zeros_like(evals)
    for ie in range (n):
        evals2[ie]=evals[idx[ie]]
        evecs2[:,ie]=1.0*evecs[:,idx[ie]]
    return evals2,evecs2

def _A_imp(u1, u2, Nl):
    mat1 = numpy.array([u1(l) for l in range(Nl)])
    mat2 = numpy.array([u2(l) for l in range(Nl)])
    return numpy.einsum('i,j->ij', mat1, mat2)

class ThreePointGFBasis(object):
    def __init__(self, Lambda, beta, Nl, path, orthogonalize = False, cutoff = 1e-8):
        self._Lambda = Lambda
        self._beta = beta
        self._Nl = Nl
        self._bf = irbasis.load('F', Lambda)
        self._bb = irbasis.load('B', Lambda)
        assert Nl <= self._bf.dim()
        assert Nl <= self._bb.dim()
        print("dim ", self._bf.dim(), self._bb.dim())
        self._scale = numpy.sqrt(2/self._beta)
        self._sqrt_beta = numpy.sqrt(self._beta)

        if orthogonalize:
            S = _overlap_matrix(self._ultauf, self._ultaub, self._beta, Nl, deg=20)
    
            evals, evecs = _eigh_ordered(S)
            evals = evals[::-1]
            evecs = evecs[:,::-1]
    
            self._Nl_ortho = numpy.sum(evals/evals[0] > cutoff)
    
            self._ortho_basis = evecs[:, 0:self._Nl_ortho]
            for l in range(self._Nl_ortho):
                self._ortho_basis[:, l] /= numpy.sqrt(evals[l])

        self._Tnl_f_cache = {}
        self._Tnl_b_cache = {}

    def ortho_basis(self):
        return self._ortho_basis

    def projector_tau(self, tau1, tau2, tau3):
        tau13 = tau1 - tau3
        tau23 = tau2 - tau3
        M = numpy.zeros((3, self._Nl, self._Nl))
        M[0, :, :] = _A_imp(lambda l: self._ultauf(l, tau13), lambda l: self._ultauf(l, tau23), self._Nl)
        M[1, :, :] = _A_imp(lambda l: self._ultaub(l, tau13), lambda l: self._ultauf(l, tau13 + tau23), self._Nl)
        M[2, :, :] = _A_imp(lambda l: self._ultaub(l, tau23), lambda l: self._ultauf(l, tau13 + tau23), self._Nl)
        return M

    def projector_iwn(self, n1, n2):
        M = numpy.zeros((3, self._Nl, self._Nl), dtype=complex)
        M[0, :, :] = numpy.einsum('i,j->ij', self._get_Tnl_f(n1),    self._get_Tnl_f(n2))
        M[1, :, :] = numpy.einsum('i,j->ij', self._get_Tnl_b(n1-n2), self._get_Tnl_f(n2))
        M[2, :, :] = numpy.einsum('i,j->ij', self._get_Tnl_b(n2-n1), self._get_Tnl_f(n1))
        M *= self._sqrt_beta**3
        return M

    def evaluate_tau(self, coeff, tau1, tau2, tau3):
        M = self.projector_tau(tau1, tau2, tau3)
        return numpy.dot(M.reshape((1, 3 * self._Nl**2)), coeff.reshape((3 * self._Nl**2, 1))) [0,0]

    def evaluate_iwn(self, coeff, n1, n2):
        M = self.projector_iwn(n1, n2)
        return numpy.dot(M.reshape((1, 3 * self._Nl**2)), coeff.reshape((3 * self._Nl**2, 1))) [0,0]

    def precompute_Tnl_b(self, n_vec):
        num_n = len(n_vec)
        Tnl_b = self._bb.compute_Tnl(n_vec)[:,0:self._Nl].reshape((num_n, self._Nl))
        for i in range(num_n):
            self._Tnl_b_cache[n_vec[i]] = Tnl_b[i,:]

    def precompute_Tnl_f(self, n_vec):
        num_n = len(n_vec)
        Tnl_f = self._bf.compute_Tnl(n_vec)[:,0:self._Nl].reshape((num_n, self._Nl))
        for i in range(num_n):
            self._Tnl_f_cache[n_vec[i]] = Tnl_f[i,:]

    def _get_Tnl_f(self, n):
        if not n in self._Tnl_f_cache:
            self._Tnl_f_cache[n] = self._bf.compute_Tnl(numpy.array([n]))[:,0:self._Nl].reshape((self._Nl))
        return self._Tnl_f_cache[n]

    def _get_Tnl_b(self, n):
        if not n in self._Tnl_b_cache:
            self._Tnl_b_cache[n] = self._bb.compute_Tnl(numpy.array([n]))[:,0:self._Nl].reshape((self._Nl))
        return self._Tnl_b_cache[n]

    def _ultauf(self, l, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        assert -1 < x and x < 1
        return s * self._bf.ulx(l, x) * self._scale

    def _ultaub(self, l, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        assert -1 < x and x < 1
        if l == 0:
            return numpy.sqrt(0.5) * self._scale
        elif l == 1:
            return numpy.sqrt(1.5) * x * self._scale
        else:
            return self._bb.ulx(l, x) * self._scale

def _scale_leggauss(x, w, x_min, x_max):
    dx = x_max - x_min
    return (dx/2)*(x+1)+x_min, w*(dx/2)

def _composite_legguass(deg, section_edges):
    x_loc,w_loc = numpy.polynomial.legendre.leggauss(deg)

    ns = len(section_edges)-1
    x = []
    w = []
    for s in range(ns):
        dx = section_edges[s+1] - section_edges[s]
        x0 = section_edges[s]
        x.extend( ((dx/2)*(x_loc+1)+x0).tolist() )
        w.extend( (w_loc*(dx/2)).tolist() )

    return numpy.array(x), numpy.array(w)

def _generate_2D_weights(deg, section_edges):
    x_template, w_template = _composite_legguass(deg, section_edges)

    points_2D = []
    w_2D = []
    for ix in range(len(x_template)):
        # [-1, x] and [x, 1]
        x = x_template[ix]
        for yr in [[-1, -x], [-x, 1]]:
            y_tmp, w_tmp = _scale_leggauss(x_template, w_template, yr[0], yr[1])
            for y in y_tmp:
                points_2D.append((x,y))
            w_2D.extend((w_template[ix]*w_tmp).tolist())
    return points_2D, numpy.array(w_2D)

def _overlap_matrix(uf, ub, beta, Nl, deg):
    # We work out with x coordinate
    ufx = lambda x : uf(Nl-1, 0.5 * beta * (x + 1) )
    zeros_f = _find_zeros(ufx)
    sections = numpy.array([-1.0] + zeros_f.tolist() + [1.0])
    print(sections)
    p_2D, w_2D = _generate_2D_weights(deg, sections)
    Np = len(p_2D)

    # Transform from x to tau coordinate
    p_2D = 0.5 * beta * (numpy.array(p_2D) + 1)
    w_2D *= (0.5 * beta) ** 2

    def make_cache(cache, u, tau, Nl):
        if not tau in cache:
            cache[tau] = numpy.array([u(l,tau) for l in range(Nl)])

    uf_cache = {}
    ub_cache = {}
    for p in range(Np):
        tau1 = p_2D[p][0]
        tau2 = p_2D[p][1]
        make_cache(uf_cache, uf, tau1, Nl)
        make_cache(uf_cache, uf, tau2, Nl)
        make_cache(uf_cache, uf, tau1+tau2, Nl)
        make_cache(ub_cache, ub, tau1, Nl)
        make_cache(ub_cache, ub, tau2, Nl)

    M = numpy.zeros((Np, 3, Nl, Nl))
    for p in range(Np):
        tau1 = p_2D[p][0]
        tau2 = p_2D[p][1]
        assert tau1 != tau2
        M[p, 0, :, :] = numpy.einsum('i,j->ij', uf_cache[tau1], uf_cache[tau2])
        M[p, 1, :, :] = numpy.einsum('i,j->ij', ub_cache[tau1], uf_cache[tau1+tau2])
        M[p, 2, :, :] = numpy.einsum('i,j->ij', ub_cache[tau2], uf_cache[tau1+tau2])

    M = M.reshape((Np,3*Nl*Nl))
    return numpy.dot(numpy.einsum('ij,j->ij', M.T, w_2D), M)
